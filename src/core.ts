import { Backend } from "./backend";
import { erf, erfc, erfinv, randInt, randNormal, randUniform } from "./utils";

export type TensorValue = number | TensorValue[];

export interface TensorOptions {
    shape?: readonly number[];
    strides?: readonly number[];
    grad?: Tensor;
    requiresGrad?: boolean;
    gradFn?: Function;
    children?: Tensor[];
    device?: string;
}

export class Tensor {
    public value: number[] | number;
    public readonly shape: readonly number[];
    public readonly strides: readonly number[];
    public grad?: Tensor;
    public requiresGrad: boolean;
    public gradFn: Function;
    public children: Tensor[];
    public device: string;

    constructor(value: TensorValue, options: TensorOptions = {}) {
        this.value = Tensor.flatten(value);
        this.shape = options.shape || Tensor.getShape(value);
        this.strides = options.strides || Tensor.getStrides(this.shape);
        this.grad = options.grad;
        this.requiresGrad = options.requiresGrad ?? false;
        this.gradFn = options.gradFn || (() => { });
        this.children = options.children || [];
        this.device = options.device || "cpu";

        // Move tensor to device
        if (this.device !== "cpu") {
            const backend = Tensor.backends.get(this.device);

            if (backend && backend.transfer) {
                backend.transfer(this);
            }
        }
    }

    // Utility to flatten an nD array to be 1D
    static flatten(tensor: TensorValue): number[] | number {
        // Handle scalar tensors
        if (typeof tensor === "number") return tensor;
        // If value is already 1D, we just need to return the value ('s reference)
        if (typeof tensor[0] === "number") return tensor as number[];

        // Or else recursively traverse through the nD array to flatten
        const result: number[] = [];

        function traverse(arr: TensorValue) {
            if (typeof arr === "number") {
                result.push(arr);
            } else if (Array.isArray(arr)) {
                arr.forEach(traverse);
            }
        }

        traverse(tensor);

        return result;
    }

    // Utility to get shape from tensor *value*
    static getShape(tensor: TensorValue): readonly number[] {
        const shape: number[] = [];

        let subA = tensor;

        while (Array.isArray(subA)) {
            shape.push(subA.length);
            subA = subA[0];
        }

        return shape;
    }

    // Utility to get strides from shape
    static getStrides(shape: readonly number[]): readonly number[] {
        if (shape.length === 0) return [];

        const strides: number[] = new Array(shape.length);

        strides[strides.length - 1] = 1;

        for (let i = strides.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        return strides;
    }

    // Left-pad shape and strides for two shape to be of same length
    static padShape(
        stridesA: readonly number[],
        stridesB: readonly number[],
        shapeA: readonly number[],
        shapeB: readonly number[]
    ): [
            readonly number[],
            readonly number[],
            readonly number[],
            readonly number[]
        ] {
        const newStrideA = [...stridesA], newStrideB = [...stridesB];
        const newShapeA = [...shapeA], newShapeB = [...shapeB];

        while (newStrideA.length < newStrideB.length) {
            const newStride = newShapeA[0] * newStrideA[0];
            newStrideA.unshift(newStride);
            newShapeA.unshift(1);
        }

        while (newStrideA.length > newStrideB.length) {
            const newStride = newShapeB[0] * newStrideB[0];
            newStrideB.unshift(newStride);
            newShapeB.unshift(1);
        }

        return [newStrideA, newStrideB, newShapeA, newShapeB];
    }

    // Broadcast shapes
    static broadcastShapes(shapeA: readonly number[], shapeB: readonly number[]): readonly number[] {
        const newShape = new Array(shapeA.length);

        for (let index = 0; index < shapeA.length; index++) {
            if (shapeA[index] === 1) {
                newShape[index] = shapeB[index];
            } else if (shapeB[index] === 1) {
                newShape[index] = shapeA[index]
            } else if (shapeA[index] === shapeB[index]) {
                newShape[index] = shapeA[index]
            } else {
                throw new Error(`Cannot broadcast shapes: ${shapeA} and ${shapeB}`);
            }
        }

        return newShape;
    }

    // Utility to convert flat index to array of coordinates
    static indexToCoords(index: number, strides: readonly number[]): number[] {
        const coords = new Array(strides.length);
        let remaining = index;

        for (let dim = 0; dim < strides.length; dim++) {
            coords[dim] = Math.floor(remaining / strides[dim]);
            remaining %= strides[dim];
        }

        return coords;
    }

    // Utility to convert array of coordinates to *unbroadcasted* flat index 
    static coordsToUnbroadcastedIndex(coords: number[], shape: readonly number[], strides: readonly number[]): number {
        let index = 0;

        for (let i = 0; i < coords.length; i++) {
            // Handle broadcasting
            const actualCoord = shape[i] === 1 ? 0 : coords[i];
            index += actualCoord * strides[i];
        }

        return index;
    }

    // Utility to convert array of coordinates to flat index 
    static coordsToIndex(coords: number[], strides: readonly number[]): number {
        let index = 0;

        for (let i = 0; i < coords.length; i++) {
            index += coords[i] * strides[i];
        }

        return index;
    }

    // Utility to convert shape into 1D value array size
    static shapeToSize(shape: readonly number[]): number {
        let prod = 1;

        for (let i = 0; i < shape.length; i++) {
            prod *= shape[i];
        }

        return prod;
    };

    // Utility for binary (two operators involved) element-wise ops
    static elementWiseAB(tA: Tensor, tB: Tensor, op: (tA: number, tB: number) => number): Tensor {
        if (typeof tA.value === "number" && typeof tB.value === "number") {
            return new Tensor(op(tA.value, tB.value));
        }

        if (typeof tA.value === "number") {
            return Tensor.elementWiseSelf(tB, (a) => op(a, tA.value as number));
        }

        if (typeof tB.value === "number") {
            return Tensor.elementWiseSelf(tA, (a) => op(a, tB.value as number));
        }

        // Pad + broadcast shape
        const [paddedAStrides, paddedBStrides, paddedAShape, paddedBShape] = Tensor.padShape(tA.strides, tB.strides, tA.shape, tB.shape);
        const outputShape = Tensor.broadcastShapes(paddedAShape, paddedBShape);
        // Get other output info
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue: number[] = new Array(outputSize);

        for (let i = 0; i < outputSize; i++) {
            // Get coordinates from 1D index
            const coordsOutput = Tensor.indexToCoords(i, outputStrides);
            // Convert the coordinates to 1D index of flattened A with respect to A's shape
            const indexA = Tensor.coordsToUnbroadcastedIndex(coordsOutput, paddedAShape, paddedAStrides);
            // Convert the coordinates to 1D index of flattened B with respect to B's shape
            const indexB = Tensor.coordsToUnbroadcastedIndex(coordsOutput, paddedBShape, paddedBStrides);

            // Calculate with op
            outputValue[i] = op((tA.value as number[])[indexA], (tB.value as number[])[indexB]);
        }

        return new Tensor(outputValue, {
            shape: outputShape,
            strides: outputStrides
        });
    }

    // Utility for self-inflicting element-wise ops
    static elementWiseSelf(tA: Tensor, op: (tA: number) => number): Tensor {
        if (typeof tA.value === "number") return new Tensor(op(tA.value));

        const newValue = new Array(tA.value.length);

        for (let index = 0; index < tA.value.length; index++) {
            newValue[index] = op(tA.value[index]);
        }

        return new Tensor(newValue, { shape: tA.shape, strides: tA.strides });
    }

    // Utility to do element-wise operation and build a dag node with another tensor
    elementWiseABDAG(
        other: TensorValue | Tensor,
        op: (a: number, b: number) => number,
        thisGrad: (self: Tensor, other: Tensor, outGrad: Tensor) => Tensor = () => new Tensor(0),
        otherGrad: (self: Tensor, other: Tensor, outGrad: Tensor) => Tensor = () => new Tensor(0)
    ): Tensor {
        other = Tensor.forceTensor(other);

        const out = Tensor.elementWiseAB(this, other, op);

        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
        }

        if (other.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(other);
        }

        if (out.requiresGrad) {
            out.gradFn = () => {
                // Disable gradient collecting of gradients themselves
                const outGrad = (out.grad as Tensor).withGrad(false);
                const selfNoGrad = this.withGrad(false);
                const otherNoGrad = other.withGrad(false);

                if (this.requiresGrad) Tensor.addGrad(this, thisGrad(selfNoGrad, otherNoGrad, outGrad));
                if (other.requiresGrad) Tensor.addGrad(other, otherGrad(selfNoGrad, otherNoGrad, outGrad));
            };
        }

        return out;
    }

    // Utility to do self-inflicting element-wise operation and build a dag node
    elementWiseSelfDAG(
        op: (a: number) => number,
        thisGrad: (self: Tensor, outGrad: Tensor) => Tensor = () => new Tensor(0)
    ): Tensor {
        const out = Tensor.elementWiseSelf(this, op);

        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
        }

        if (out.requiresGrad) {
            out.gradFn = () => {
                // Disable gradient collecting of gradients themselves
                const outGrad = (out.grad as Tensor).withGrad(false);
                const selfNoGrad = this.withGrad(false);

                if (this.requiresGrad) Tensor.addGrad(this, thisGrad(selfNoGrad, outGrad));
            };
        }

        return out;
    }

    // Utility to force an input value to be a tensor
    static forceTensor(value: TensorValue | Tensor): Tensor {
        if (value instanceof Tensor) return value;
        return new Tensor(value);
    }

    // Utility to add to gradient of tensor
    static addGrad(tensor: Tensor, accumGrad: Tensor) {
        const axesToSqueeze = [];
        const axesToReduce = [];

        const shape = tensor.shape;
        const gradShape = accumGrad.shape;

        const paddedDims = gradShape.length - shape.length;

        for (let i = 0; i < paddedDims; i++) {
            axesToReduce.push(i);
            axesToSqueeze.push(i);
        }

        for (let i = 0; i < shape.length; i++) {
            if (shape[i] === 1 && gradShape[i + paddedDims] > 1) {
                axesToReduce.push(i + paddedDims);
            }
        }

        const reducedGrad = accumGrad.sum(axesToReduce, true);
        const squeezedGrad = reducedGrad.squeeze(axesToSqueeze);

        if (typeof tensor.grad === "undefined") {
            tensor.grad = squeezedGrad;
        } else {
            tensor.grad = tensor.grad.add(squeezedGrad);
        }
    }

    // Tensor squeeze
    squeeze(dims?: number[] | number): Tensor {
        if (typeof this.value === "number") return this;
        if (typeof dims === "number") { dims = [dims]; }
        if (typeof dims === "undefined") {
            const shape = this.shape;

            dims = [];

            for (let index = 0; index < shape.length; index++) {
                if (shape[index] === 1) {
                    dims.push(index);
                }
            }
        }

        // Remove size-1 dims only
        const outShape: number[] = [], outStrides: number[] = [];

        for (let index = 0; index < this.shape.length; index++) {
            const dim = this.shape[index];
            const stride = this.strides[index];

            if (dims.includes(index)) {
                if (dim !== 1) throw new Error(`Can not squeeze dim with size ${dim}`);
            } else {
                outShape.push(dim);
                outStrides.push(stride);
            }
        }

        const outValue = outShape.length === 0 ? this.value[0] : this.value;
        const out = new Tensor(outValue, {
            shape: outShape,
            strides: outStrides,
            device: this.device
        });

        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                let restoredGrad = (out.grad as Tensor).withGrad(false);

                for (let i = dims.length - 1; i >= 0; i--) {
                    restoredGrad = restoredGrad.unsqueeze(dims[i]);
                }

                Tensor.addGrad(this, restoredGrad);
            };
        }

        return out;
    }

    // Tensor unsqueeze - adds dimension of size 1 at specified position
    unsqueeze(dim: number): Tensor {
        let thisValue = this.value;

        if (typeof thisValue === "number") {
            thisValue = [thisValue];
        }

        // Insert size-1 dimension at specified position
        const newShape = [...this.shape];
        newShape.splice(dim, 0, 1);

        // New stride
        const newStrides = [...this.strides];
        let newDimStride;
        if (dim >= this.shape.length) {
            // Inserting at the back: use 1
            newDimStride = 1;
        } else {
            // Inserting before dim: use current stride * current shape
            newDimStride = this.strides[dim] * this.shape[dim];
        }
        newStrides.splice(dim, 0, newDimStride);

        const out = new Tensor(thisValue, { shape: newShape, strides: newStrides, device: this.device });

        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, (out.grad as Tensor).withGrad(false).squeeze(dim));
            };
        }

        return out;
    }

    // Tensor sum reduction
    sum(dims?: number[] | number, keepDims: boolean = false): Tensor {
        if (typeof this.value === "number") return this;

        if (typeof dims === "undefined") { dims = Array.from({ length: this.shape.length }, (_, index) => index); }

        if (Array.isArray(dims)) {
            // Sort in descending order
            const sortedDims = dims.sort((a, b) => b - a);

            let reducedThis: Tensor = this;

            for (let i = 0; i < sortedDims.length; i++) {
                reducedThis = reducedThis.sum(sortedDims[i], true);
            }

            return keepDims ? reducedThis : reducedThis.squeeze(dims);
        }

        // Dims that are reduced now have size-1
        const outputShape = this.shape.map((dim, i) => dims === i ? 1 : dim);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize).fill(0);
        const originalSize = Tensor.shapeToSize(this.shape);

        // Gradient data
        let gradShape: readonly number[], gradStrides: readonly number[], gradValue: number[] = [];
        // Allocate gradient data only when needed
        if (this.requiresGrad) {
            gradShape = this.shape;
            gradStrides = this.strides;
            gradValue = new Array(originalSize).fill(0);
        }

        // Calculate new value after sum
        for (let realFlatIndex = 0; realFlatIndex < originalSize; realFlatIndex++) {
            const coords = Tensor.indexToCoords(realFlatIndex, this.strides);
            // Force 0 on reduced axes to collapse into size-1 dims
            const outCoords = coords.map((val, i) => dims === i ? 0 : val);
            // Convert output coordinates to flat index
            const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
            // Add into sum
            outputValue[outFlatIndex] += this.value[realFlatIndex];
            // Mark for gradient if needed
            if (this.requiresGrad) { gradValue[realFlatIndex] = 1; }
        }

        const out = new Tensor(outputValue, {
            shape: outputShape,
            strides: outputStrides
        });

        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                const localGrad = new Tensor(gradValue, { shape: gradShape, strides: gradStrides });
                Tensor.addGrad(this, (out.grad as Tensor).withGrad(false).mul(localGrad));
            };
        }

        return keepDims ? out : out.squeeze(dims);
    }

    // Tensor product reduction
    prod(dims?: number[] | number, keepDims: boolean = false): Tensor {
        if (typeof this.value === "number") return this;

        if (typeof dims === "undefined") { dims = Array.from({ length: this.shape.length }, (_, index) => index); }

        if (Array.isArray(dims)) {
            // Sort in descending order
            const sortedDims = dims.sort((a, b) => b - a);

            let reducedThis: Tensor = this;

            for (let i = 0; i < sortedDims.length; i++) {
                reducedThis = reducedThis.prod(sortedDims[i], true);
            }

            return keepDims ? reducedThis : reducedThis.squeeze(dims);
        }

        // Dims that are reduced now have size-1
        const outputShape = this.shape.map((dim, i) => dims === i ? 1 : dim);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize).fill(1);
        const originalSize = Tensor.shapeToSize(this.shape);

        // Calculate new value after multiplying
        for (let realFlatIndex = 0; realFlatIndex < originalSize; realFlatIndex++) {
            const coords = Tensor.indexToCoords(realFlatIndex, this.strides);
            // Force 0 on reduced axes to collapse into size-1 dims
            const outCoords = coords.map((val, i) => dims === i ? 0 : val);
            // Convert output coordinates to flat index
            const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
            // Multiply into product
            outputValue[outFlatIndex] *= this.value[realFlatIndex];
        }

        const out = new Tensor(outputValue, {
            shape: outputShape,
            strides: outputStrides
        });

        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                const gradShape = this.shape, gradStrides = this.strides, gradValue = new Array(originalSize).fill(0);

                for (let realFlatIndex = 0; realFlatIndex < originalSize; realFlatIndex++) {
                    const coords = Tensor.indexToCoords(realFlatIndex, this.strides);
                    // Force 0 on reduced axes to collapse into size-1 dims
                    const outCoords = coords.map((val, i) => dims === i ? 0 : val);
                    // Convert output coordinates to flat index
                    const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
                    // Grad is the product of other elements of the same axis, which is product of all els divided by the current value
                    gradValue[realFlatIndex] = outputValue[outFlatIndex] / (this.value as number[])[realFlatIndex];
                }

                const localGrad = new Tensor(gradValue, { shape: gradShape, strides: gradStrides });
                Tensor.addGrad(this, (out.grad as Tensor).withGrad(false).mul(localGrad));
            };
        }

        return keepDims ? out : out.squeeze(dims);
    }

    // Tensor mean reduction
    mean(dims?: number[] | number, keepDims: boolean = false): Tensor {
        if (typeof this.value === "number") return this;

        if (typeof dims === "undefined") { dims = Array.from({ length: this.shape.length }, (_, index) => index); }

        if (Array.isArray(dims)) {
            // Sort in descending order
            const sortedDims = dims.sort((a, b) => b - a);

            let reducedThis: Tensor = this;

            for (let i = 0; i < sortedDims.length; i++) {
                reducedThis = reducedThis.mean(sortedDims[i], true);
            }

            return keepDims ? reducedThis : reducedThis.squeeze(dims);
        }

        // Dims that are reduced now have size-1
        const outputShape = this.shape.map((dim, i) => dims === i ? 1 : dim);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize).fill(0);
        const outputFeeders = new Array(outputSize).fill(0);
        const originalSize = Tensor.shapeToSize(this.shape);

        // Calculate sums and how many elements contribute to specific positions
        for (let realFlatIndex = 0; realFlatIndex < originalSize; realFlatIndex++) {
            const coords = Tensor.indexToCoords(realFlatIndex, this.strides);
            // Force 0 on reduced axes to collapse into size-1 dims
            const outCoords = coords.map((val, i) => dims === i ? 0 : val);
            // Convert output coordinates to flat index
            const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
            // Calculate sum and contributors to the sum
            outputValue[outFlatIndex] += this.value[realFlatIndex];
            outputFeeders[outFlatIndex]++;
        }

        // Calculate mean by dividing sum by the number of contributors to the position
        for (let index = 0; index < outputSize; index++) {
            outputValue[index] /= outputFeeders[index];
        }

        const out = new Tensor(outputValue, {
            shape: outputShape,
            strides: outputStrides
        });

        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                const gradShape = this.shape, gradStrides = this.strides, gradValue = new Array(originalSize).fill(0);

                // Calculate grad by assigning 1 divided by the number of contributors to the position
                for (let realFlatIndex = 0; realFlatIndex < originalSize; realFlatIndex++) {
                    const coords = Tensor.indexToCoords(realFlatIndex, this.strides);
                    // Force 0 on reduced axes to collapse into size-1 dims
                    const outCoords = coords.map((val, i) => dims === i ? 0 : val);
                    // Convert output coordinates to flat index
                    const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
                    // Mean = 1/n * (el1 + el2 + ... + eln) so grad = 1/n
                    gradValue[realFlatIndex] = 1 / outputFeeders[outFlatIndex];
                }

                const localGrad = new Tensor(gradValue, { shape: gradShape, strides: gradStrides });
                Tensor.addGrad(this, (out.grad as Tensor).withGrad(false).mul(localGrad));
            };
        }

        return keepDims ? out : out.squeeze(dims);
    }

    // Tensor maximum reduction
    max(dims?: number[] | number, keepDims: boolean = false): Tensor {
        if (typeof this.value === "number") return this;

        if (typeof dims === "undefined") { dims = Array.from({ length: this.shape.length }, (_, index) => index); }

        if (Array.isArray(dims)) {
            // Sort in descending order
            const sortedDims = dims.sort((a, b) => b - a);

            let reducedThis: Tensor = this;

            for (let i = 0; i < sortedDims.length; i++) {
                reducedThis = reducedThis.max(sortedDims[i], true);
            }

            return keepDims ? reducedThis : reducedThis.squeeze(dims);
        }

        // Dims that are reduced now have size-1
        const outputShape = this.shape.map((dim, i) => dims === i ? 1 : dim);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize).fill(-Infinity);
        const originalSize = Tensor.shapeToSize(this.shape);

        // Calculate maximum values of axes
        for (let realFlatIndex = 0; realFlatIndex < originalSize; realFlatIndex++) {
            const coords = Tensor.indexToCoords(realFlatIndex, this.strides);
            // Force 0 on reduced axes to collapse into size-1 dims
            const outCoords = coords.map((val, i) => dims === i ? 0 : val);
            // Convert output coordinates to flat index
            const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
            // Get max over time
            if (this.value[realFlatIndex] > outputValue[outFlatIndex]) {
                outputValue[outFlatIndex] = this.value[realFlatIndex];
            }
        }

        const out = new Tensor(outputValue, {
            shape: outputShape,
            strides: outputStrides
        });

        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                const gradShape = this.shape, gradStrides = this.strides, gradValue = new Array(originalSize).fill(0);
                const shareCounts = new Array(outputSize).fill(0);
                const originalValue = this.value as number[];

                for (let realFlatIndex = 0; realFlatIndex < originalSize; realFlatIndex++) {
                    const coords = Tensor.indexToCoords(realFlatIndex, this.strides);
                    // Force 0 on reduced axes to collapse into size-1 dims
                    const outCoords = coords.map((val, i) => dims === i ? 0 : val);
                    // Convert output coordinates to flat index
                    const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
                    // We collect how many elements share the same max value first
                    shareCounts[outFlatIndex] += outputValue[outFlatIndex] === originalValue[realFlatIndex] ? 1 : 0;
                }

                for (let realFlatIndex = 0; realFlatIndex < originalSize; realFlatIndex++) {
                    const coords = Tensor.indexToCoords(realFlatIndex, this.strides);
                    // Force 0 on reduced axes to collapse into size-1 dims
                    const outCoords = coords.map((val, i) => dims === i ? 0 : val);
                    // Convert output coordinates to flat index
                    const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
                    // Here we share the grad between the elements that share the same max value
                    gradValue[realFlatIndex] = outputValue[outFlatIndex] === originalValue[realFlatIndex] ? 1 / shareCounts[outFlatIndex] : 0; 
                }

                const localGrad = new Tensor(gradValue, { shape: gradShape, strides: gradStrides });
                Tensor.addGrad(this, (out.grad as Tensor).withGrad(false).mul(localGrad));
            };
        }

        return keepDims ? out : out.squeeze(dims);
    }

    // Tensor minimum reduction
    min(dims?: number[] | number, keepDims: boolean = false): Tensor {
        if (typeof this.value === "number") return this;

        if (typeof dims === "undefined") { dims = Array.from({ length: this.shape.length }, (_, index) => index); }

        if (Array.isArray(dims)) {
            // Sort in descending order
            const sortedDims = dims.sort((a, b) => b - a);

            let reducedThis: Tensor = this;

            for (let i = 0; i < sortedDims.length; i++) {
                reducedThis = reducedThis.min(sortedDims[i], true);
            }

            return keepDims ? reducedThis : reducedThis.squeeze(dims);
        }

        // Dims that are reduced now have size-1
        const outputShape = this.shape.map((dim, i) => dims === i ? 1 : dim);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize).fill(Infinity);
        const originalSize = Tensor.shapeToSize(this.shape);

        // Calculate minimum values of axes
        for (let realFlatIndex = 0; realFlatIndex < originalSize; realFlatIndex++) {
            const coords = Tensor.indexToCoords(realFlatIndex, this.strides);
            // Force 0 on reduced axes to collapse into size-1 dims
            const outCoords = coords.map((val, i) => dims === i ? 0 : val);
            // Convert output coordinates to flat index
            const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
            // Get min over time
            if (this.value[realFlatIndex] < outputValue[outFlatIndex]) {
                outputValue[outFlatIndex] = this.value[realFlatIndex];
            }
        }

        const out = new Tensor(outputValue, {
            shape: outputShape,
            strides: outputStrides
        });

        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                const gradShape = this.shape, gradStrides = this.strides, gradValue = new Array(originalSize).fill(0);
                const shareCounts = new Array(outputSize).fill(0);
                const originalValue = this.value as number[];

                for (let realFlatIndex = 0; realFlatIndex < originalSize; realFlatIndex++) {
                    const coords = Tensor.indexToCoords(realFlatIndex, this.strides);
                    // Force 0 on reduced axes to collapse into size-1 dims
                    const outCoords = coords.map((val, i) => dims === i ? 0 : val);
                    // Convert output coordinates to flat index
                    const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
                    // We collect how many elements share the same min value first
                    shareCounts[outFlatIndex] += outputValue[outFlatIndex] === originalValue[realFlatIndex] ? 1 : 0;
                }

                for (let realFlatIndex = 0; realFlatIndex < originalSize; realFlatIndex++) {
                    const coords = Tensor.indexToCoords(realFlatIndex, this.strides);
                    // Force 0 on reduced axes to collapse into size-1 dims
                    const outCoords = coords.map((val, i) => dims === i ? 0 : val);
                    // Convert output coordinates to flat index
                    const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
                    // Here we share the grad between the elements that share the same min value
                    gradValue[realFlatIndex] = outputValue[outFlatIndex] === originalValue[realFlatIndex] ? 1 / shareCounts[outFlatIndex] : 0; 
                }

                const localGrad = new Tensor(gradValue, { shape: gradShape, strides: gradStrides });
                Tensor.addGrad(this, (out.grad as Tensor).withGrad(false).mul(localGrad));
            };
        }

        return keepDims ? out : out.squeeze(dims);
    }

    // Tensor variance reduction
    var(dims?: number[] | number, keepDims: boolean = false): Tensor {
        const meanXSquared = this.square().mean(dims, keepDims);
        const meanXSquaredExpanded = this.mean(dims, keepDims).square();

        return meanXSquared.sub(meanXSquaredExpanded);
    }

    // Tensor standard deviation reduction
    std(dims?: number[] | number, keepDims: boolean = false): Tensor {
        return this.var(dims, keepDims).sqrt();
    }

    // Tensor product reduction
    softmax(dims?: number[] | number): Tensor {
        if (typeof this.value === "number") return this;

        if (typeof dims === "undefined") { dims = Array.from({ length: this.shape.length }, (_, index) => index); }

        if (Array.isArray(dims)) {
            // Sort in descending order
            const sortedDims = dims.sort((a, b) => b - a);

            let reducedThis: Tensor = this;

            for (let i = 0; i < sortedDims.length; i++) {
                reducedThis = reducedThis.softmax(sortedDims[i]);
            }

            return reducedThis;
        }

        // Dims that are reduced now have size-1
        const expSumShape = this.shape.map((dim, i) => dims === i ? 1 : dim);
        const expSumStrides = Tensor.getStrides(expSumShape);
        const expSumSize = Tensor.shapeToSize(expSumShape);
        const expSumValue = new Array(expSumSize).fill(0);
        const outputShape = this.shape;
        const outputStrides = this.strides;
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize);

        // Calculate sums of e^xi over axes
        for (let realFlatIndex = 0; realFlatIndex < outputSize; realFlatIndex++) {
            const coords = Tensor.indexToCoords(realFlatIndex, outputStrides);
            // Force 0 on reduced axes to collapse into size-1 dims
            const expSumCoords = coords.map((val, i) => dims === i ? 0 : val);
            // Convert exp sum coordinates to flat index
            const expSumFlatIndex = Tensor.coordsToIndex(expSumCoords, expSumStrides);
            // Add e^x to the sum cache
            expSumValue[expSumFlatIndex] += Math.exp(this.value[realFlatIndex]);
        }

        // Calculate e^xi / sum over axes
        for (let realFlatIndex = 0; realFlatIndex < outputSize; realFlatIndex++) {
            const coords = Tensor.indexToCoords(realFlatIndex, outputStrides);
            // Force 0 on reduced axes to collapse into size-1 dims
            const expSumCoords = coords.map((val, i) => dims === i ? 0 : val);
            // Convert exp sum coordinates to flat index
            const expSumFlatIndex = Tensor.coordsToIndex(expSumCoords, expSumStrides);
            // Calculate e^xi / sum
            outputValue[realFlatIndex] = Math.exp(this.value[realFlatIndex]) / expSumValue[expSumFlatIndex];
        }

        const out = new Tensor(outputValue, {
            shape: outputShape,
            strides: outputStrides
        });

        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                const upstreamGrad = (out.grad as Tensor).withGrad(false);
                const softmaxOutput = out.withGrad(false);

                // Compute element-wise product: ∂L/∂σᵢ × σᵢ
                const gradTimesOutput = upstreamGrad.mul(softmaxOutput);

                // Sum over softmax dimensions: Σᵢ(∂L/∂σᵢ × σᵢ)
                const sumGradOutput = gradTimesOutput.sum(dims, true); // keepDims=true for broadcasting

                // Apply softmax gradient formula:
                // ∂L/∂zⱼ = (∂L/∂σⱼ × σⱼ) - (σⱼ × Σᵢ(∂L/∂σᵢ × σᵢ))
                const term1 = upstreamGrad.mul(softmaxOutput);  // ∂L/∂σⱼ × σⱼ
                const term2 = softmaxOutput.mul(sumGradOutput); // σⱼ × Σᵢ(∂L/∂σᵢ × σᵢ)
                const localGrad = term1.sub(term2);

                Tensor.addGrad(this, localGrad);
            };
        }

        return out;
    }

    // Tensor element-wise addition
    add(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a + b,
            (self, other, outGrad) => outGrad,
            (self, other, outGrad) => outGrad
        );
    }

    // Tensor element-wise subtraction
    sub(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a - b,
            (self, other, outGrad) => outGrad,
            (self, other, outGrad) => outGrad.neg()
        );
    }

    subtract = this.sub;

    // Tensor element-wise multiplication
    mul(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a * b,
            (self, other, outGrad) => outGrad.mul(other),
            (self, other, outGrad) => outGrad.mul(self)
        );
    }

    multiply = this.mul;

    // Tensor element-wise power
    pow(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a ** b,
            (self, other, outGrad) => outGrad.mul(other.mul(self.pow(other.sub(1)))),
            (self, other, outGrad) => outGrad.mul(self.pow(other).mul(self.log()))
        );
    }

    // Tensor element-wise division
    div(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a / b,
            (self, other, outGrad) => outGrad.div(other),
            (self, other, outGrad) => outGrad.mul(self.neg().div(other.square()))
        );
    }

    divide = this.div;

    // Tensor element-wise modulo
    remainder(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a % b
        );
    }

    // Tensor element-wise greater or equal comparison
    ge(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a >= b ? 1 : 0
        );
    }

    greaterEqual = this.ge;

    // Tensor element-wise less or equal comparison
    le(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a <= b ? 1 : 0
        );
    }

    lessEqual = this.le;

    // Tensor element-wise greater-than comparison
    gt(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a > b ? 1 : 0
        );
    }

    greater = this.gt;

    // Tensor element-wise less-than comparison
    lt(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a < b ? 1 : 0
        );
    }

    less = this.lt;

    // Tensor element-wise equality comparison
    eq(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a === b ? 1 : 0
        );
    }

    equal = this.eq;

    // Tensor element-wise not equality comparison
    ne(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a !== b ? 1 : 0
        );
    }

    notEqual = this.ne;

    // Tensor element-wise logical and
    logicalAnd(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a === 1 && b === 1 ? 1 : 0
        );
    }

    // Tensor element-wise logical or
    logicalOr(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a === 1 || b === 1 ? 1 : 0
        );
    }

    // Tensor element-wise logical xor
    logicalXor(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => (a === 1 || b === 1) && a !== b ? 1 : 0
        );
    }

    // Tensor element-wise logical not
    logicalNot(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => a === 1 ? 0 : 1
        );
    }

    // Tensor element-wise bitwise and
    bitwiseAnd(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a & b
        );
    }

    // Tensor element-wise bitwise or
    bitwiseOr(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a | b
        );
    }

    // Tensor element-wise bitwise xor
    bitwiseXor(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a ^ b
        );
    }

    // Tensor element-wise bitwise not
    bitwiseNot(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => ~a
        );
    }

    // Tensor element-wise left shift
    bitwiseLeftShift(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a << b
        );
    }

    // Tensor element-wise right shift
    bitwiseRightShift(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => a >> b
        );
    }

    // Tensor element-wise negation
    neg(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => -a,
            (self, outGrad) => outGrad.mul(-1)
        );
    }

    negative = this.neg;

    // Tensor element-wise reciprocal
    reciprocal(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => 1 / a,
            (self, outGrad) => outGrad.mul(self.pow(-2).neg())
        );
    }

    // Tensor element-wise square
    square(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => a * a,
            (self, outGrad) => outGrad.mul(self.mul(2))
        );
    }

    // Tensor element-wise absolute
    abs(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.abs(a),
            (self, outGrad) => outGrad.mul(self.sign())
        );
    }

    absolute = this.abs;

    // Tensor element-wise sign function
    sign(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.sign(a)
        );
    }

    // Tensor element-wise sin
    sin(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.sin(a),
            (self, outGrad) => outGrad.mul(self.cos())
        );
    }

    // Tensor element-wise cos
    cos(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.cos(a),
            (self, outGrad) => outGrad.mul(self.sin().neg())
        );
    }

    // Tensor element-wise tan
    tan(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.tan(a),
            (self, outGrad) => outGrad.mul(self.tan().square().add(1))
        );
    }

    // Tensor element-wise asin
    asin(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.asin(a),
            (self, outGrad) => outGrad.div(self.square().neg().add(1).sqrt())
        );
    }

    arcsin = this.asin;

    // Tensor element-wise acos
    acos(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.acos(a),
            (self, outGrad) => outGrad.div(self.square().neg().add(1).sqrt()).neg()
        );
    }

    arccos = this.acos;

    // Tensor element-wise atan
    atan(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.atan(a),
            (self, outGrad) => outGrad.div(self.square().add(1))
        );
    }

    arctan = this.atan;

    // Tensor element-wise atan2
    atan2(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => Math.atan2(a, b),
            (self, other, outGrad) => outGrad.mul(other.div(self.square().add(other.square()))),
            (self, other, outGrad) => outGrad.mul(self.neg().div(self.square().add(other.square())))
        );
    }

    arctan2 = this.atan2;

    // Tensor element-wise sinh
    sinh(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.sinh(a),
            (self, outGrad) => outGrad.mul(self.cosh())
        );
    }

    // Tensor element-wise cosh
    cosh(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.cosh(a),
            (self, outGrad) => outGrad.mul(self.sinh())
        );
    }

    // Tensor element-wise asinh
    asinh(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.asinh(a),
            (self, outGrad) => outGrad.div(self.square().add(1).sqrt())
        );
    }

    arcsinh = this.asinh;

    // Tensor element-wise acosh
    acosh(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.acosh(a),
            (self, outGrad) => outGrad.div(self.add(1).sqrt().mul(self.sub(1).sqrt()))
        );
    }

    arccosh = this.acosh;

    // Tensor element-wise atanh
    atanh(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.atanh(a),
            (self, outGrad) => outGrad.div(self.square().neg().add(1))
        );
    }

    arctanh = this.atanh;

    // Tensor element-wise degree to radian
    deg2rad(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => a * (Math.PI / 180),
            (self, outGrad) => outGrad.mul(Math.PI / 180)
        );
    }

    // Tensor element-wise radian to degree
    rad2deg(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => a / (Math.PI / 180),
            (self, outGrad) => outGrad.div(Math.PI / 180)
        );
    }

    // Tensor element-wise square root
    sqrt(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.sqrt(a),
            (self, outGrad) => outGrad.div(self.sqrt().mul(2))
        );
    }

    // Tensor element-wise reciprocal of square root
    rsqrt(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => 1 / Math.sqrt(a),
            (self, outGrad) => outGrad.mul(self.pow(-1.5).mul(-0.5))
        );
    }

    // Tensor element-wise e^x
    exp(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.exp(a),
            (self, outGrad) => outGrad.mul(self.exp())
        );
    }

    // Tensor element-wise 2^x
    exp2(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => 2 ** a,
            (self, outGrad) => outGrad.mul(self.exp2().mul(Math.log(2)))
        );
    }

    // Tensor element-wise e^x - 1
    expm1(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.expm1(a),
            (self, outGrad) => outGrad.mul(self.exp())
        );
    }

    // Tensor element-wise natural log
    log(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.log(a),
            (self, outGrad) => outGrad.div(self)
        );
    }

    // Tensor element-wise log2
    log2(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.log2(a),
            (self, outGrad) => outGrad.div(self.mul(Math.log(2)))
        );
    }

    // Tensor element-wise log10
    log10(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.log10(a),
            (self, outGrad) => outGrad.div(self.mul(Math.log(10)))
        );
    }

    // Tensor element-wise log(1+x)
    log1p(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.log1p(a),
            (self, outGrad) => outGrad.div(self.add(1))
        );
    }

    // Tensor element-wise relu
    relu(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.max(a, 0),
            (self, outGrad) => outGrad.mul(self.gt(0))
        );
    }

    // Tensor element-wise sigmoid
    sigmoid(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => 1 / (1 + Math.exp(-a)),
            (self, outGrad) => {
                const sig = self.sigmoid();
                return outGrad.mul(sig).mul(sig.neg().add(1));
            }
        );
    }

    // Tensor element-wise tanh
    tanh(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.tanh(a),
            (self, outGrad) => outGrad.mul(self.tanh().square().neg().add(1))
        );
    }

    // Tensor element-wise softplus
    softplus(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.log1p(Math.exp(a)),
            (self, outGrad) => outGrad.mul(self.sigmoid())
        );
    }

    // Tensor element-wise softsign
    softsign(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => a / (1 + Math.abs(a)),
            (self, outGrad) => outGrad.div(self.abs().add(1).square())
        );
    }

    // Tensor element-wise silu (swish)
    silu(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => a / (1 + Math.exp(-a)),
            (self, outGrad) => {
                const sig = self.sigmoid();
                return outGrad.mul(sig.add(self.mul(sig).mul(sig.neg().add(1))));
            }
        );
    }

    // Tensor element-wise mish
    mish(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => a * Math.tanh(Math.log1p(Math.exp(a))),
            (self, outGrad) => {
                const tanhSoftPlus = self.exp().add(1).log().tanh();

                // tanh(softplus(x)) + x * (1 - tanh²(softplus(x))) * sigmoid(x)
                const derivative = tanhSoftPlus.add(
                    self.mul(tanhSoftPlus.square().neg().add(1)).mul(self.sigmoid())
                );

                return outGrad.mul(derivative);
            }
        );
    }

    // Tensor element-wise maximum
    maximum(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => Math.max(a, b),
            (self, other, outGrad) => outGrad.mul(self.gt(other).add(self.eq(other).mul(0.5))),
            (self, other, outGrad) => outGrad.mul(other.gt(self).add(other.eq(self).mul(0.5)))
        );
    }

    // Tensor element-wise minimum
    minimum(other: TensorValue | Tensor): Tensor {
        return this.elementWiseABDAG(
            other,
            (a, b) => Math.min(a, b),
            (self, other, outGrad) => outGrad.mul(self.lt(other).add(self.eq(other).mul(0.5))),
            (self, other, outGrad) => outGrad.mul(other.lt(self).add(other.eq(self).mul(0.5)))
        );
    }

    // Tensor element-wise round
    round(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.round(a)
        );
    }

    // Tensor element-wise floor
    floor(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.floor(a)
        );
    }

    // Tensor element-wise ceil
    ceil(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.ceil(a)
        );
    }

    // Tensor element-wise truncation
    trunc(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.trunc(a)
        );
    }

    fix = this.trunc;

    // Tensor element-wise fraction portion
    frac(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => a - Math.floor(a)
        );
    }

    // Tensor element-wise clip and clamp
    clip(min: number, max: number): Tensor {
        return this.elementWiseSelfDAG(
            (a) => Math.max(min, Math.min(max, a)),
            (self, outGrad) => outGrad.mul(self.ge(min).mul(self.le(max)))
        );
    }

    clamp = this.clip;

    // Tensor element-wise error function
    erf(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => erf(a),
            (self, outGrad) => outGrad.mul(self.square().neg().exp().mul(2 / Math.sqrt(Math.PI)))
        );
    }

    // Tensor element-wise complementary error function
    erfc(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => erfc(a),
            (self, outGrad) => outGrad.mul(self.square().neg().exp().mul(2 / Math.sqrt(Math.PI)).neg())
        );
    }

    // Tensor element-wise inverse error function
    erfinv(): Tensor {
        return this.elementWiseSelfDAG(
            (a) => erfinv(a),
            (self, outGrad) => outGrad.mul(self.erfinv().square().exp().mul(Math.sqrt(Math.PI) / 2))
        );
    }

    // Transpose
    transpose(dim1: number, dim2: number): Tensor {
        // If dimension out of bound, throw error
        if (dim1 >= this.shape.length || dim2 >= this.shape.length || dim1 < 0 || dim2 < 0) {
            throw new Error("Dimensions do not exist to tranpose");
        }

        // If same dimension, return copy
        if (dim1 === dim2) {
            return new Tensor(this.value, { shape: this.shape, strides: this.strides });
        }

        // Create new shape and strides by swapping
        const newShape = [...this.shape];
        const newStrides = [...this.strides];

        [newShape[dim1], newShape[dim2]] = [newShape[dim2], newShape[dim1]];
        [newStrides[dim1], newStrides[dim2]] = [newStrides[dim2], newStrides[dim1]];

        // Create new tensor with same data but swapped shape/strides
        const out = new Tensor(this.value, { shape: newShape, strides: newStrides, device: this.device });
        out.requiresGrad = this.requiresGrad;

        // Handle gradient if needed
        if (this.requiresGrad) {
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, (out.grad as Tensor).withGrad(false).transpose(dim1, dim2));
            };
        }

        return out;
    }

    swapaxes = this.transpose;
    swapdims = this.transpose;

    // Transpose 2D
    t(): Tensor {
        // Verify matrix shape
        if (this.shape.length !== 2) {
            throw new Error("Input is not a matrix");
        }

        return this.transpose(0, 1);
    }

    // 1D tensor dot product
    dot(other: TensorValue | Tensor): Tensor {
        other = Tensor.forceTensor(other);

        // Verify 1D shape
        if (this.shape.length !== 1 || other.shape.length !== 1) {
            throw new Error("Inputs are not 1D tensors");
        }

        // Simple vector dot product
        const vectLen = this.shape[0];
        const vectA = this.value as number[];
        const vectB = other.value as number[];
        let sum = 0;

        for (let index = 0; index < vectLen; index++) {
            sum += vectA[index] * vectB[index];
        }

        const out = new Tensor(sum);

        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
        }

        if (other.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(other);
        }

        if (out.requiresGrad) {
            out.gradFn = () => {
                // Disable gradient collecting of gradients themselves
                const outGrad = (out.grad as Tensor).withGrad(false);
                const selfNoGrad = this.withGrad(false);
                const otherNoGrad = other.withGrad(false);

                if (this.requiresGrad) Tensor.addGrad(this, outGrad.mul(otherNoGrad))
                if (other.requiresGrad) Tensor.addGrad(other, outGrad.mul(selfNoGrad));
            };
        }

        return out;
    }

    // Matrix multiplication
    mm(other: TensorValue | Tensor): Tensor {
        other = Tensor.forceTensor(other);

        // Verify 2D shape
        if (this.shape.length !== 2 || other.shape.length !== 2) {
            throw new Error("Inputs are not matrices");
        }

        // Simple matrix multiplication
        const matA = this.value as number[];
        const matB = other.value as number[];
        const matAStrides = this.strides;
        const matBStrides = other.strides;
        const matARows = this.shape[0];
        const matACols = this.shape[1];
        const matBRows = other.shape[0];
        const matBCols = other.shape[1];

        if (matACols !== matBRows) throw new Error("Invalid matrices shape for multiplication");

        const matCShape = [matARows, matBCols];
        const matCStrides = Tensor.getStrides(matCShape);
        const matCSize = Tensor.shapeToSize(matCShape);
        const matC = new Array(matCSize).fill(0);

        for (let i = 0; i < matARows; i++) {
            for (let j = 0; j < matBCols; j++) {
                for (let k = 0; k < matACols; k++) {
                    // Tensor values are 1D arrays so we have to get real index using strides
                    matC[i * matCStrides[0] + j * matCStrides[1]] +=
                        matA[i * matAStrides[0] + k * matAStrides[1]] *
                        matB[k * matBStrides[0] + j * matBStrides[1]];
                }
            }
        }

        const out = new Tensor(matC, { shape: matCShape, strides: matCStrides });

        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
        }

        if (other.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(other);
        }

        if (out.requiresGrad) {
            out.gradFn = () => {
                // Disable gradient collecting of gradients themselves
                const outGrad = (out.grad as Tensor).withGrad(false);
                const selfNoGrad = this.withGrad(false);
                const otherNoGrad = other.withGrad(false);

                if (this.requiresGrad) Tensor.addGrad(this, outGrad.mm(otherNoGrad.t()));
                if (other.requiresGrad) Tensor.addGrad(other, selfNoGrad.t().mm(outGrad));
            };
        }

        return out;
    }

    // Batched 3D tensor matmul
    bmm(other: TensorValue | Tensor): Tensor {
        other = Tensor.forceTensor(other);

        // Verify 3D shape
        if (this.shape.length !== 3 || other.shape.length !== 3 || this.shape[0] !== other.shape[0]) {
            throw new Error("Inputs are not 3D tensors with the same first dim size");
        }

        // Simple matrix multiplication
        const batchA = this.value as number[];
        const batchB = other.value as number[];
        const batchAStrides = this.strides;
        const batchBStrides = other.strides;
        const batchSize = this.shape[0];
        const batchARows = this.shape[1];
        const batchACols = this.shape[2];
        const batchBRows = other.shape[1];
        const batchBCols = other.shape[2];

        if (batchACols !== batchBRows) throw new Error("Invalid matrices shape for multiplication");

        const batchCShape = [batchSize, batchARows, batchBCols];
        const batchCStrides = Tensor.getStrides(batchCShape);
        const batchCSize = Tensor.shapeToSize(batchCShape);
        const batchC = new Array(batchCSize).fill(0);

        for (let q = 0; q < batchSize; q++) {
            for (let i = 0; i < batchARows; i++) {
                for (let j = 0; j < batchBCols; j++) {
                    for (let k = 0; k < batchACols; k++) {
                        // Tensor values are 1D arrays so we have to get real index using strides
                        batchC[q * batchCStrides[0] + i * batchCStrides[1] + j * batchCStrides[2]] +=
                            batchA[q * batchAStrides[0] + i * batchAStrides[1] + k * batchAStrides[2]] *
                            batchB[q * batchBStrides[0] + k * batchBStrides[1] + j * batchBStrides[2]];
                    }
                }
            }
        }

        const out = new Tensor(batchC, { shape: batchCShape, strides: batchCStrides });

        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
        }

        if (other.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(other);
        }

        if (out.requiresGrad) {
            out.gradFn = () => {
                // Disable gradient collecting of gradients themselves
                const outGrad = (out.grad as Tensor).withGrad(false);
                const selfNoGrad = this.withGrad(false);
                const otherNoGrad = other.withGrad(false);

                if (this.requiresGrad) Tensor.addGrad(this, outGrad.bmm(otherNoGrad.transpose(1, 2)));
                if (other.requiresGrad) Tensor.addGrad(other, selfNoGrad.transpose(1, 2).bmm(outGrad));
            };
        }

        return out;
    }

    // Convert right-side 1D tensor to a vector (nx1 tensor) to do matmul
    mv(other: TensorValue | Tensor): Tensor {
        other = Tensor.forceTensor(other);

        // Verify 2D shape
        if (this.shape.length !== 2 || other.shape.length !== 1) {
            throw new Error("Input is not a 2D and 1D tensor pair");
        }

        return this.mm(other.unsqueeze(1)).squeeze(1);
    }

    // General matrix multiplication with different shapes
    matmul(other: TensorValue | Tensor): Tensor {
        other = Tensor.forceTensor(other);

        const isThis1D = this.shape.length === 1;
        const isOther1D = other.shape.length === 1;

        if (isThis1D && isOther1D) {
            return this.dot(other);
        } else if (isThis1D && other.shape.length === 2) {
            return this.unsqueeze(0).mm(other).squeeze(0);
        } else if (this.shape.length === 2 && isOther1D) {
            return this.mv(other);
        } else if (this.shape.length === 2 && other.shape.length === 2) {
            return this.mm(other);
        } else if (
            (isThis1D && other.shape.length > 2) ||
            (isOther1D && this.shape.length > 2) ||
            (other.shape.length > 2 && this.shape.length > 2)
        ) {
            // Append/prepend dims if needed
            const self = isThis1D ? this.unsqueeze(0) : this;
            other = isOther1D ? other.unsqueeze(1) : other;

            // Padding
            const [selfStrides, otherStrides, selfShape, otherShape] = Tensor.padShape(self.strides, other.strides, self.shape, other.shape);
            const lastDim = selfShape.length - 1;

            // Prepare data for broadcasting
            const batchA = self.value as number[];
            const batchB = other.value as number[];
            const batchARows = selfShape[lastDim - 1];
            const batchACols = selfShape[lastDim];
            const batchBRows = otherShape[lastDim - 1];
            const batchBCols = otherShape[lastDim];

            // Verify if can do matmul
            if (batchACols !== batchBRows) throw new Error("Invalid matrices shape for multiplication");

            // Prepare shape, strides, size info, but more importantly the offset-related data to loop through the outer, non-matrix dims
            // Self and other's offset data
            const selfOffsetShape = selfShape.slice(0, -2);
            const otherOffsetShape = otherShape.slice(0, -2);
            const selfOffsetStrides = selfStrides.slice(0, -2);
            const otherOffsetStrides = otherStrides.slice(0, -2);
            // The output's offset data
            const offsetShape = Tensor.broadcastShapes(selfOffsetShape, otherOffsetShape);
            const offsetSize = Tensor.shapeToSize(offsetShape);
            const offsetStrides = Tensor.getStrides(offsetShape);
            // Output shape, strides, size, value
            const outputShape = [...offsetShape, batchARows, batchBCols];
            const outputStrides = Tensor.getStrides(outputShape);
            const outputSize = Tensor.shapeToSize(outputShape);
            const outputValue = new Array(outputSize).fill(0);

            // Loop through outer dims and do matmul on two outer-most dims
            for (let index = 0; index < offsetSize; index++) {
                const coords = Tensor.indexToCoords(index, offsetStrides);
                const offset = Tensor.coordsToIndex(coords, outputStrides.slice(0, -2));
                const selfOffset = Tensor.coordsToUnbroadcastedIndex(coords, selfOffsetShape, selfOffsetStrides);
                const otherOffset = Tensor.coordsToUnbroadcastedIndex(coords, otherOffsetShape, otherOffsetStrides);

                for (let i = 0; i < batchARows; i++) {
                    for (let j = 0; j < batchBCols; j++) {
                        for (let k = 0; k < batchACols; k++) {
                            const outputIdx = offset + i * outputStrides[lastDim - 1] + j * outputStrides[lastDim];
                            const selfIdx = selfOffset + i * selfStrides[lastDim - 1] + k * selfStrides[lastDim];
                            const otherIdx = otherOffset + k * otherStrides[lastDim - 1] + j * otherStrides[lastDim];

                            outputValue[outputIdx] += batchA[selfIdx] * batchB[otherIdx];
                        }
                    }
                }
            }

            const out = new Tensor(outputValue, { shape: outputShape, strides: outputStrides });

            if (this.requiresGrad) {
                out.requiresGrad = true;
                out.children.push(this);
            }

            if (other.requiresGrad) {
                out.requiresGrad = true;
                out.children.push(other);
            }

            if (out.requiresGrad) {
                out.gradFn = () => {
                    other = other as Tensor;
                    const outGrad = (out.grad as Tensor).withGrad(false);
                    const selfNoGrad = self.withGrad(false);
                    const otherNoGrad = other.withGrad(false);

                    if (this.requiresGrad) Tensor.addGrad(this, outGrad.matmul(otherNoGrad.transpose(lastDim - 1, lastDim)));
                    if (other.requiresGrad) Tensor.addGrad(other, selfNoGrad.transpose(lastDim - 1, lastDim).matmul(outGrad));
                };
            }

            return out;
        }

        throw new Error(`Shapes [${this.shape}] and [${other.shape}] are not supported`);
    }

    // Utility to create a new tensor filled with a number
    static full(shape: number[], num: number, options: TensorOptions = {}): Tensor {
        if (shape.length === 0) return new Tensor(num, options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize).fill(num);

        return new Tensor(outputValue, { shape, ...options })
    }

    // Utility to create a new tensor with shape of another tensor, filled with a number
    static fullLike(tensor: Tensor, num: number, options: TensorOptions = {}): Tensor {
        if (typeof tensor.value === "number") return new Tensor(num, options);

        return new Tensor(new Array(tensor.value.length).fill(num), { shape: tensor.shape, strides: tensor.strides, ...options });
    }

    // Utility to create a new tensor filled with 1
    static ones(shape?: number[], options: TensorOptions = {}): Tensor {
        if (typeof shape === "undefined" || shape.length === 0) return new Tensor(1, options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize).fill(1);

        return new Tensor(outputValue, { shape, ...options })
    }

    // Utility to create a new tensor with shape of another tensor, filled with 1
    static onesLike(tensor: Tensor, options: TensorOptions = {}): Tensor {
        if (typeof tensor.value === "number") return new Tensor(1, options);

        return new Tensor(new Array(tensor.value.length).fill(1), { shape: tensor.shape, strides: tensor.strides, ...options });
    }

    // Utility to create a new tensor filled with 0
    static zeros(shape?: number[], options: TensorOptions = {}): Tensor {
        if (typeof shape === "undefined" || shape.length === 0) return new Tensor(0, options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize).fill(0);

        return new Tensor(outputValue, { shape, ...options })
    }

    // Utility to create a new tensor with shape of another tensor, filled with 0
    static zerosLike(tensor: Tensor, options: TensorOptions = {}): Tensor {
        if (typeof tensor.value === "number") return new Tensor(0, options);

        return new Tensor(new Array(tensor.value.length).fill(0), { shape: tensor.shape, strides: tensor.strides, ...options });
    }

    // Utility to create a new tensor filled with a random number with uniform distribution from 0 to 1
    static rand(shape?: number[], options: TensorOptions = {}): Tensor {
        if (typeof shape === "undefined" || shape.length === 0) return new Tensor(randUniform(), options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randUniform();
        }

        return new Tensor(outputValue, { shape, ...options })
    }

    // Utility to create a new tensor with shape of another tensor, filled with a random number with uniform distribution from 0 to 1
    static randLike(tensor: Tensor, options: TensorOptions = {}): Tensor {
        if (typeof tensor.value === "number") return new Tensor(randUniform(), options);

        const outputValue = new Array(tensor.value.length);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randUniform();
        }

        return new Tensor(
            outputValue,
            {
                shape: tensor.shape, strides: tensor.strides, ...options
            }
        );
    }

    // Utility to create a new tensor filled with a random number with normal distribution of mean=0 and stddev=1
    static randn(shape?: number[], options: TensorOptions = {}): Tensor {
        if (typeof shape === "undefined" || shape.length === 0) return new Tensor(randNormal(), options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randNormal();
        }

        return new Tensor(outputValue, { shape, ...options })
    }

    // Utility to create a new tensor with shape of another tensor, filled with a random number with normal distribution of mean=0 and stddev=1
    static randnLike(tensor: Tensor, options: TensorOptions = {}): Tensor {
        if (typeof tensor.value === "number") return new Tensor(randNormal(), options);

        const outputValue = new Array(tensor.value.length);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randNormal();
        }

        return new Tensor(
            outputValue,
            {
                shape: tensor.shape, strides: tensor.strides, ...options
            }
        );
    }

    // Utility to create a new tensor filled with a random integer between low and high
    static randint(shape: number[], low: number, high: number, options: TensorOptions = {}): Tensor {
        if (shape.length === 0) return new Tensor(randInt(low, high), options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randInt(low, high);
        }

        return new Tensor(outputValue, { shape, ...options })
    }

    // Utility to create a new tensor with shape of another tensor, filled with a random integer between low and high
    static randintLike(tensor: Tensor, low: number, high: number, options: TensorOptions = {}): Tensor {
        if (typeof tensor.value === "number") return new Tensor(randInt(low, high), options);

        const outputValue = new Array(tensor.value.length);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randInt(low, high);
        }

        return new Tensor(
            outputValue,
            {
                shape: tensor.shape, strides: tensor.strides, ...options
            }
        );
    }

    // Utility to create a new tensor filled with a random number with normal distribution of custom mean and stddev
    static normal(shape: number[], mean: number, stdDev: number, options: TensorOptions = {}): Tensor {
        if (shape.length === 0) return new Tensor(randNormal(mean, stdDev), options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randNormal(mean, stdDev);
        }

        return new Tensor(outputValue, { shape, ...options })
    }

    // Utility to create a new tensor filled with a random number with uniform distribution from low to high
    static uniform(shape: number[], low: number, high: number, options: TensorOptions = {}): Tensor {
        if (shape.length === 0) return new Tensor(randUniform(low, high), options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randUniform(low, high);
        }

        return new Tensor(outputValue, { shape, ...options })
    }

    // Reverse-mode autodiff call
    backward() {
        // Build topological order
        const topo: Tensor[] = [];
        const visited: Set<Tensor> = new Set();

        function build(node: Tensor) {
            if (!visited.has(node) && node.requiresGrad) {
                visited.add(node);
                node.grad = Tensor.zerosLike(node); // Reset grad with 0
                for (let child of node.children) build(child);
                topo.push(node);
            }
        }

        build(this);

        // Feed backward to calculate gradient
        this.grad = Tensor.onesLike(this);

        for (let index = topo.length - 1; index > -1; index--) {
            topo[index].gradFn();
        }
    }

    // Returns the raw number/nD array form of tensor
    val(): TensorValue {
        if (typeof this.value === "number") return this.value;

        function buildNested(
            data: number[],
            shape: readonly number[],
            strides: readonly number[],
            baseIndex = 0,
            dim = 0
        ): TensorValue {
            if (dim === shape.length - 1) {
                // Last dimension: extract elements using actual stride
                const result = [];

                for (let i = 0; i < shape[dim]; i++) {
                    result.push(data[baseIndex + i * strides[dim]]);
                }

                return result;
            }

            // Recursive case: build nested structure
            const result = [];

            for (let i = 0; i < shape[dim]; i++) {
                result.push(buildNested(data, shape, strides, baseIndex + i * strides[dim], dim + 1));
            }

            return result;
        }

        return buildNested(this.value, this.shape, this.strides);
    }

    // Returns a view of the tensor with gradient turned on/off and detaches from autograd
    withGrad(requiresGrad: boolean): Tensor {
        return new Tensor(this.value, {
            shape: this.shape,
            strides: this.strides,
            device: this.device,
            requiresGrad
        })
    }

    // Returns a view of the tensor with gradient turned off and detaches from autograd
    detach(): Tensor {
        return new Tensor(this.value, {
            shape: this.shape,
            strides: this.strides,
            device: this.device,
            requiresGrad: false
        })
    }

    // Returns a copy of the tensor (with new data allocation) and detaches from autograd
    clone(): Tensor {
        return new Tensor(typeof this.value === "number" ? this.value : [...this.value], {
            shape: this.shape,
            strides: this.strides,
            requiresGrad: this.requiresGrad
        })
    }

    // Returns this tensor with value replaced with the value of another tensor
    replace(other: Tensor, allowShapeMismatch: boolean = false): Tensor {
        // Verify shape
        if (!allowShapeMismatch) {
            for (let index = 0; index < this.shape.length; index++) {
                if (this.shape[index] !== other.shape[index]) {
                    throw new Error("Shape mismatch when trying to do tensor value replacement");
                }
            }
        }

        this.value = other.value;

        return this;
    }

    // Holds all available backends
    static backends: Map<string, Backend> = new Map();

    // Op to transfer tensor to another device
    to(device: string): Tensor {
        const backend = Tensor.backends.get(device);

        if (backend && backend.transfer) {
            backend.transfer(this);
            return this;
        }

        throw new Error(`No device found to transfer tensor to or a handler is not implemented for device.`);
    }
}

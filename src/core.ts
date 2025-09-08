import { Backend } from "./backend";
import { erf, erfc, erfinv, fyShuffle, randInt, randNormal, randUniform } from "./utils";

export type TensorValue = number | TensorValue[];

export interface TensorOptions {
    shape?: readonly number[];
    strides?: readonly number[];
    offset?: number;
    numel?: number;
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
    public offset: number;
    public numel: number;
    public grad?: Tensor;
    public requiresGrad: boolean;
    public gradFn: Function;
    public children: Tensor[];
    public device: string;
    static training: boolean = false;

    constructor(value: TensorValue, options: TensorOptions = {}) {
        // Storage
        this.value = Tensor.flatten(value);

        // Tensor metadata
        this.shape = options.shape || Tensor.getShape(value);
        this.strides = options.strides || Tensor.getStrides(this.shape);
        this.offset = options.offset || 0;
        this.numel = options.numel || Tensor.shapeToSize(this.shape);
        this.device = options.device || "cpu";

        // Autograd data
        this.grad = options.grad;
        this.requiresGrad = options.requiresGrad ?? false;
        this.gradFn = options.gradFn || (() => { });
        this.children = options.children || [];

        // Move to device in-place
        this.to_(this.device);
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
                throw new Error(`Can not broadcast shapes: ${shapeA} and ${shapeB}`);
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
            outputValue[i] = op(tA.value[indexA + tA.offset], tB.value[indexB + tB.offset]);
        }

        return new Tensor(outputValue, {
            shape: outputShape,
            strides: outputStrides,
            numel: outputSize
        });
    }

    // Utility for self-inflicting element-wise ops
    static elementWiseSelf(tA: Tensor, op: (tA: number) => number): Tensor {
        if (typeof tA.value === "number") return new Tensor(op(tA.value));

        const outputShape = tA.shape;
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = tA.numel;
        const outputValue: number[] = new Array(outputSize);

        for (let index = 0; index < outputSize; index++) {
            const outputCoords = Tensor.indexToCoords(index, outputStrides);
            const originalIndex = tA.offset + Tensor.coordsToIndex(outputCoords, tA.strides);
            outputValue[index] = op(tA.value[originalIndex]);
        }

        return new Tensor(outputValue, { shape: outputShape, strides: outputStrides, numel: tA.numel });
    }

    // Utility to do element-wise operation and build a dag node with another tensor
    elementWiseABDAG(
        other: TensorValue | Tensor,
        op: (a: number, b: number) => number,
        thisGrad: (self: Tensor, other: Tensor, outGrad: Tensor) => Tensor = () => new Tensor(0),
        otherGrad: (self: Tensor, other: Tensor, outGrad: Tensor) => Tensor = () => new Tensor(0)
    ): Tensor {
        other = this.handleOther(other);

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
                const outGrad = (out.grad as Tensor);
                const selfNoGrad = this.detach();
                const otherNoGrad = other.detach();

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
                const outGrad = (out.grad as Tensor);
                const selfNoGrad = this.detach();

                if (this.requiresGrad) Tensor.addGrad(this, thisGrad(selfNoGrad, outGrad));
            };
        }

        return out;
    }

    // Utility to handle other tensor if an op needs a second operand
    handleOther(other: Tensor | TensorValue): Tensor {
        if (other instanceof Tensor) {
            if (this.device !== other.device) {
                throw new Error("Can not operate on tensors that are not on the same device");
            }

            return other;
        }

        return new Tensor(other, { device: this.device });
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

    static normalizeDims(dims: number[], numDims: number): number[] {
        for (let index = 0; index < dims.length; index++) {
            // Handle negative indices
            if (dims[index] < 0) {
                dims[index] += numDims;
            }

            // If dimension out of bound, throw error
            if (dims[index] >= numDims || dims[index] < 0) {
                throw new Error("Dimensions do not exist");
            }
        }

        return dims;
    }

    // Contiguity-related ops
    isContiguous(): boolean {
        const expectedStrides = Tensor.getStrides(this.shape);

        if (expectedStrides.length !== this.strides.length) {
            return false;
        }

        for (let i = 0; i < this.strides.length; i++) {
            if (this.strides[i] !== expectedStrides[i]) {
                return false;
            }
        }

        return true;
    }

    contiguous(): Tensor {
        // Check if scalar
        if (typeof this.value === "number") return this;
        // Check if already contiguous
        if (this.isContiguous()) return this;

        const outputStrides = Tensor.getStrides(this.shape);
        const outputSize = this.numel;
        const outputValue = new Array(outputSize);

        for (let index = 0; index < outputSize; index++) {
            const outputCoords = Tensor.indexToCoords(index, outputStrides);
            const originalIndex = Tensor.coordsToIndex(outputCoords, this.strides);

            outputValue[index] = this.value[this.offset + originalIndex];
        }

        const out = new Tensor(outputValue, { shape: this.shape, strides: outputStrides, numel: outputSize })

        // Gradient flow back to the original tensor
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, (out.grad as Tensor));
            };
        }

        return out;
    }

    view(newShape: readonly number[]): Tensor {
        // Verify shape size
        const originalSize = this.numel;
        const outputSize = Tensor.shapeToSize(newShape);

        if (originalSize !== outputSize) {
            throw new Error("Can not create view: incompatible sizes");
        }

        // Verify compatibility (only contiguity for now)
        if (!this.isContiguous()) {
            throw new Error("Can not create view: incompatible metadata");
        }

        const outputStrides = Tensor.getStrides(newShape);
        const out = new Tensor(this.value, { shape: newShape, strides: outputStrides, numel: outputSize });

        // Gradient reshaped and flow back to the original tensor
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, (out.grad as Tensor).reshape(this.shape));
            };
        }

        return out;
    }

    reshape(newShape: readonly number[]): Tensor {
        // Verify shape size
        const originalSize = this.numel;
        const outputSize = Tensor.shapeToSize(newShape);

        if (originalSize !== outputSize) {
            throw new Error("Can not reshape: incompatible sizes");
        }

        // Create new tensor with forced compatibility (only contiguity for now)
        const outputStrides = Tensor.getStrides(newShape);
        const out = new Tensor(this.contiguous().value, { shape: newShape, strides: outputStrides, numel: outputSize });

        // Gradient reshaped and flow back to the original tensor
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, (out.grad as Tensor).reshape(this.shape));
            };
        }

        return out;
    }

    // Transpose
    transpose(dim1: number, dim2: number): Tensor {
        // Handle negative indices
        if (dim1 < 0) { dim1 += this.shape.length }
        if (dim2 < 0) { dim2 += this.shape.length }

        // If dimension out of bound, throw error
        if (dim1 >= this.shape.length || dim2 >= this.shape.length || dim1 < 0 || dim2 < 0) {
            throw new Error("Dimensions do not exist to transpose");
        }

        // If same dimension, return view
        if (dim1 === dim2) return this;

        // Create new shape and strides by swapping
        const newShape = [...this.shape];
        const newStrides = [...this.strides];

        [newShape[dim1], newShape[dim2]] = [newShape[dim2], newShape[dim1]];
        [newStrides[dim1], newStrides[dim2]] = [newStrides[dim2], newStrides[dim1]];

        // Create new tensor with same data but swapped shape/strides
        const out = new Tensor(this.value, {
            shape: newShape,
            strides: newStrides,
            offset: this.offset,
            numel: this.numel,
            device: this.device
        });
        out.requiresGrad = this.requiresGrad;

        // Handle gradient if needed
        if (this.requiresGrad) {
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, (out.grad as Tensor).transpose(dim1, dim2));
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

    // Permute
    permute(dims: number[]): Tensor {
        dims = Tensor.normalizeDims(dims, this.shape.length);

        if (dims.length !== this.shape.length) {
            throw new Error("Permutation must specify all dimensions");
        }

        // Compute new shape and strides
        const newShape = new Array(dims.length);
        const newStrides = new Array(dims.length);

        for (let index = 0; index < dims.length; index++) {
            const dim = dims[index];
            newShape[index] = this.shape[dim];
            newStrides[index] = this.strides[dim];
        }

        const out = new Tensor(this.value, {
            shape: newShape,
            strides: newStrides,
            offset: this.offset,
            numel: this.numel,
            device: this.device
        });

        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                // Compute inverse permutation
                const inverseAxes = new Array(dims.length);
                for (let i = 0; i < dims.length; i++) {
                    inverseAxes[dims[i]] = i;
                }

                // Permute gradient back to original order
                const permutedGrad = (out.grad as Tensor).permute(inverseAxes);
                Tensor.addGrad(this, permutedGrad);
            };
        }

        return out;
    }

    // Utility for indexing with array of indices
    indexWithArray(indices: number[]): Tensor {
        if (typeof this.value === "number") return this;

        indices = Tensor.normalizeDims(indices, this.shape[0]);

        // Init necessary stuff for indexing
        const reducedShape = this.shape.slice(1);
        const reducedStrides = this.strides.slice(1);
        const elementsPerIndex = Tensor.shapeToSize(reducedShape);

        // Init output data
        const outputShape = [indices.length, ...reducedShape];
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize);

        for (let i = 0; i < indices.length; i++) {
            const sourceRowIndex = indices[i];
            const targetStart = i * elementsPerIndex;

            for (let j = 0; j < elementsPerIndex; j++) {
                const fullCoords = Tensor.indexToCoords(j, reducedStrides);
                fullCoords.unshift(sourceRowIndex);
                const sourceIndex = Tensor.coordsToIndex(fullCoords, this.strides);

                outputValue[targetStart + j] = this.value[this.offset + sourceIndex];
            }
        }

        const out = new Tensor(outputValue, {
            shape: outputShape,
            numel: outputSize
        });

        // Handle gradient
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                const outGrad = out.grad as Tensor;

                // Create zero gradient tensor with original shape
                const grad = Tensor.zerosLike(this);

                // Scatter gradients back to original positions
                for (let i = 0; i < indices.length; i++) {
                    const originalRowIndex = indices[i];
                    const sourceStart = i * elementsPerIndex;

                    for (let j = 0; j < elementsPerIndex; j++) {
                        const fullCoords = Tensor.indexToCoords(j, reducedStrides);
                        fullCoords.unshift(originalRowIndex);
                        const targetIndex = Tensor.coordsToIndex(fullCoords, this.strides);

                        (grad.value as number[])[targetIndex] += (outGrad.value as number[])[sourceStart + j];
                    }
                }

                Tensor.addGrad(this, grad);
            };
        }

        return out;
    }

    // Tensor indexing
    index(indices: Tensor | TensorValue): Tensor {
        const tensorIndices = this.handleOther(indices).contiguous();

        if (typeof tensorIndices.value === "number") {
            return this.indexWithArray([tensorIndices.value]).squeeze(0);
        } else {
            const originalShape = tensorIndices.shape;
            const flatIndices = tensorIndices.value

            const result = this.indexWithArray(flatIndices);

            // Reshape to preserve input shape
            const outputShape = [...originalShape, ...this.shape.slice(1)];
            return result.reshape(outputShape);
        }
    }

    // Tensor slicing
    slice(ranges: number[][]): Tensor {
        // Handle scalars
        if (typeof this.value === "number") return this;

        const newShape = [];
        const newStrides = [];
        let newOffset = this.offset || 0;

        // Pad ranges to match tensor dimensions
        const paddedRanges = [...ranges];
        while (paddedRanges.length < this.shape.length) {
            paddedRanges.push([]);
        }

        for (let i = 0; i < this.shape.length; i++) {
            const range = paddedRanges[i] || [];
            const dimSize = this.shape[i];
            const stride = this.strides[i];

            // Default values
            let start = range[0] ?? 0;
            let end = range[1] ?? dimSize;
            let step = range[2] ?? 1;

            // Handle negative indices
            if (start < 0) start += dimSize;
            if (end < 0) end += dimSize;

            // Clamp to valid range
            start = Math.max(0, Math.min(start, dimSize));
            end = Math.max(0, Math.min(end, dimSize));

            // Calculate new dimension size
            const newDimSize = step > 0
                ? Math.max(0, Math.ceil((end - start) / step))
                : Math.max(0, Math.ceil((start - end) / Math.abs(step)));

            newShape.push(newDimSize);
            newStrides.push(stride * step);

            newOffset += start * stride;
        }

        const out = new Tensor(this.value, {
            shape: newShape,
            strides: newStrides,
            offset: newOffset,
            device: this.device
        });

        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                // Create zero tensor of original shape
                const grad = Tensor.zerosLike(this);
                // Upstream grad
                const outGrad = out.grad as Tensor;

                const totalElements = outGrad.numel;

                for (let i = 0; i < totalElements; i++) {
                    // Convert flat index to coordinates in sliced tensor
                    const slicedCoords = Tensor.indexToCoords(i, outGrad.strides);

                    // Map back to original coordinates
                    const originalCoords = new Array(slicedCoords.length);

                    for (let dim = 0; dim < slicedCoords.length; dim++) {
                        const coord = slicedCoords[dim];
                        const range = paddedRanges[dim] || [];
                        const start = range[0] ?? 0;
                        const step = range[2] ?? 1;
                        const normalizedStart = start < 0 ? start + this.shape[dim] : start;
                        originalCoords[dim] = normalizedStart + coord * step;
                    }

                    // Get flat indices with offsets
                    const srcIndex = Tensor.coordsToIndex(slicedCoords, outGrad.strides) + outGrad.offset;
                    const targetIndex = Tensor.coordsToIndex(originalCoords, grad.strides) + grad.offset;

                    // Accumulate gradient
                    (grad.value as number[])[targetIndex] += (outGrad.value as number[])[srcIndex];
                }

                Tensor.addGrad(this, grad);
            };
        }

        return out;
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

        dims = Tensor.normalizeDims(dims, this.shape.length);

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

        const outValue = outShape.length === 0 ? this.value[this.offset] : this.value;
        const out = new Tensor(outValue, {
            shape: outShape,
            strides: outStrides,
            offset: this.offset,
            device: this.device
        });

        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                let restoredGrad = (out.grad as Tensor);

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
        // Handle negative indices
        if (dim < 0) { dim += this.shape.length }

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

        const out = new Tensor(thisValue, {
            shape: newShape,
            strides: newStrides,
            offset: this.offset,
            device: this.device
        });

        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, (out.grad as Tensor).squeeze(dim));
            };
        }

        return out;
    }

    // Generic reduction operation handler
    static reduce(
        tensor: Tensor,
        dims: number[] | number | undefined,
        keepDims: boolean,
        config: {
            identity: number;
            operation: (accumulator: number, value: number) => number;
            needsCounters?: boolean;
            postProcess?: (options: { values: number[], counters?: number[] }) => void;
            needsShareCounts?: boolean;
            gradientFn: (options: {
                outputValue: number[],
                originalValue: number[],
                counters: number[],
                shareCounts: number[],
                realIndex: number,
                outIndex: number
            }) => number;
        }
    ): Tensor {
        if (typeof tensor.value === "number") return tensor;

        if (typeof dims === "undefined") {
            dims = Array.from({ length: tensor.shape.length }, (_, index) => index);
        }

        if (Array.isArray(dims)) {
            dims = Tensor.normalizeDims(dims, tensor.shape.length);
            const sortedDims = dims.sort((a, b) => b - a);
            let reducedThis: Tensor = tensor;
            for (let i = 0; i < sortedDims.length; i++) {
                reducedThis = Tensor.reduce(reducedThis, sortedDims[i], true, config);
            }
            return keepDims ? reducedThis : reducedThis.squeeze(dims);
        }

        const outputShape = tensor.shape.map((dim, i) => dims === i ? 1 : dim);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize).fill(config.identity);
        const outputCounters = config.needsCounters ? new Array(outputSize).fill(0) : [];
        const originalSize = tensor.numel;
        const originalValue = tensor.value;
        const linearStrides = Tensor.getStrides(tensor.shape);

        // Forward pass
        for (let flatIndex = 0; flatIndex < originalSize; flatIndex++) {
            // Convert linear index to coordinates using contiguous strides
            const coords = Tensor.indexToCoords(flatIndex, linearStrides);

            // Convert coordinates to actual strided index
            const realFlatIndex = Tensor.coordsToIndex(coords, tensor.strides) + tensor.offset;

            // Convert coords to reduced index
            coords[dims] = 0;
            const outFlatIndex = Tensor.coordsToIndex(coords, outputStrides);

            // Apply op
            outputValue[outFlatIndex] = config.operation(outputValue[outFlatIndex], originalValue[realFlatIndex]);

            // Count el if needed
            if (config.needsCounters) {
                outputCounters[outFlatIndex]++;
            }
        }

        // Post-process if needed (e.g., divide by count for mean)
        if (config.postProcess) {
            config.postProcess({ values: outputValue, counters: outputCounters });
        }

        const out = new Tensor(outputValue, { shape: outputShape, strides: outputStrides, numel: outputSize });

        // Gradient setup
        if (tensor.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(tensor);
            out.gradFn = () => {
                let shareCounts = [];

                if (config.needsShareCounts) {
                    shareCounts = new Array(outputSize).fill(0);

                    for (let flatIndex = 0; flatIndex < originalSize; flatIndex++) {
                        // Convert linear index to coordinates using contiguous strides
                        const coords = Tensor.indexToCoords(flatIndex, linearStrides);

                        // Convert coordinates to actual strided index
                        const realFlatIndex = Tensor.coordsToIndex(coords, tensor.strides) + tensor.offset;

                        // Convert coords to reduced index
                        coords[dims] = 0;
                        const outFlatIndex = Tensor.coordsToIndex(coords, outputStrides);

                        // We collect how many elements share the same max value first
                        shareCounts[outFlatIndex] += outputValue[outFlatIndex] === originalValue[realFlatIndex] ? 1 : 0;
                    }
                }

                const gradValue = new Array(originalSize);

                for (let flatIndex = 0; flatIndex < originalSize; flatIndex++) {
                    // Convert linear index to coordinates using contiguous strides
                    const coords = Tensor.indexToCoords(flatIndex, linearStrides);

                    // Convert coordinates to actual strided index
                    const realFlatIndex = Tensor.coordsToIndex(coords, tensor.strides) + tensor.offset;

                    // Convert coords to reduced index
                    coords[dims] = 0;
                    const outFlatIndex = Tensor.coordsToIndex(coords, outputStrides);

                    gradValue[flatIndex] = config.gradientFn({
                        outputValue,
                        originalValue: tensor.value as number[],
                        counters: outputCounters,
                        shareCounts,
                        realIndex: realFlatIndex,
                        outIndex: outFlatIndex
                    });
                }

                const localGrad = new Tensor(gradValue, { shape: tensor.shape, numel: tensor.numel });
                Tensor.addGrad(tensor, (out.grad as Tensor).mul(localGrad));
            };
        }

        return keepDims ? out : out.squeeze(dims);
    }

    // Simplified reduction operations
    sum(dims?: number[] | number, keepDims = false): Tensor {
        return Tensor.reduce(this, dims, keepDims, {
            identity: 0,
            operation: (a, b) => a + b,
            gradientFn: ({ }) => 1
        });
    }

    prod(dims?: number[] | number, keepDims = false): Tensor {
        return Tensor.reduce(this, dims, keepDims, {
            identity: 1,
            operation: (a, b) => a * b,
            gradientFn: ({ outputValue, originalValue, realIndex, outIndex }) =>
                outputValue[outIndex] / originalValue[realIndex]
        });
    }

    mean(dims?: number[] | number, keepDims = false): Tensor {
        return Tensor.reduce(this, dims, keepDims, {
            identity: 0,
            operation: (a, b) => a + b,
            needsCounters: true,
            postProcess: ({ values, counters }) => {
                for (let i = 0; i < values.length; i++) {
                    values[i] /= counters![i];
                }
            },
            gradientFn: ({ counters, outIndex }) => 1 / counters[outIndex]
        });
    }

    max(dims?: number[] | number, keepDims = false): Tensor {
        return Tensor.reduce(this, dims, keepDims, {
            identity: -Infinity,
            operation: (a, b) => Math.max(a, b),
            needsShareCounts: true,
            gradientFn: ({ outputValue, originalValue, shareCounts, realIndex, outIndex }) =>
                outputValue[outIndex] === originalValue[realIndex] ? 1 / shareCounts[outIndex] : 0
        });
    }

    min(dims?: number[] | number, keepDims = false): Tensor {
        return Tensor.reduce(this, dims, keepDims, {
            identity: Infinity,
            operation: (a, b) => Math.min(a, b),
            needsShareCounts: true,
            gradientFn: ({ outputValue, originalValue, shareCounts, realIndex, outIndex }) =>
                outputValue[outIndex] === originalValue[realIndex] ? 1 / shareCounts[outIndex] : 0
        });
    }

    // Tensor all condition reduction
    all(dims?: number[] | number, keepDims: boolean = false): Tensor {
        return this.min(dims, keepDims).ne(0);
    }

    // Tensor any condition reduction
    any(dims?: number[] | number, keepDims: boolean = false): Tensor {
        return this.max(dims, keepDims).ne(0);
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

    // Tensor softmax
    softmax(dim: number = -1): Tensor {
        if (typeof this.value === "number") return this;

        // Handle negative indexing
        if (dim < 0) dim = this.shape.length + dim;

        const maxVals = this.max(dim, true);
        const shifted = this.sub(maxVals);
        const expVals = shifted.exp();
        const sumExp = expVals.sum(dim, true);
        return expVals.div(sumExp);
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

    // Tensor element-wise gelu
    gelu(approximate: string = "none"): Tensor {
        if (approximate === "none") {
            return this.elementWiseSelfDAG(
                (a) => 0.5 * a * (1 + erf(a / Math.sqrt(2))),
                (self, outGrad) => {
                    const sqrt2 = Math.sqrt(2);
                    const sqrt2OverPi = Math.sqrt(2 / Math.PI);

                    const xOverSqrt2 = self.div(sqrt2);
                    const erfVal = xOverSqrt2.erf();
                    const phi = xOverSqrt2.square().neg().exp().div(sqrt2OverPi);

                    const derivative = erfVal.add(1).mul(0.5).add(self.mul(phi));
                    return outGrad.mul(derivative);
                }
            );
        } else if (approximate === "tanh") {
            return this.elementWiseSelfDAG(
                (a) => 0.5 * a * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (a + 0.044715 * a * a * a))),
                (self, outGrad) => {
                    const sqrt2OverPi = Math.sqrt(2 / Math.PI);
                    const c = 0.044715;

                    const tanhArg = self.add(self.pow(3).mul(c)).mul(sqrt2OverPi);
                    const tanhVal = tanhArg.tanh();
                    const sechSquared = tanhVal.square().neg().add(1);

                    const term1 = tanhVal.add(1).mul(0.5);
                    const term2 = self.mul(sechSquared).mul(sqrt2OverPi).mul(
                        self.square().mul(c * 3).add(1)
                    ).mul(0.5);

                    const derivative = term1.add(term2);
                    return outGrad.mul(derivative);
                }
            );
        }

        throw new Error("Specified approximation does not exist");
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

    // 1D tensor dot product
    dot(other: TensorValue | Tensor): Tensor {
        other = this.handleOther(other);

        // Verify 1D shape
        if (this.shape.length !== 1 || other.shape.length !== 1) {
            throw new Error("Inputs are not 1D tensors");
        }

        return this.mul(other).sum();
    }

    // Matrix multiplication
    mm(other: TensorValue | Tensor): Tensor {
        other = this.handleOther(other);

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
                        matA[i * matAStrides[0] + k * matAStrides[1] + this.offset] *
                        matB[k * matBStrides[0] + j * matBStrides[1] + other.offset];
                }
            }
        }

        const out = new Tensor(matC, { shape: matCShape, strides: matCStrides, numel: matCSize });

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
                const outGrad = (out.grad as Tensor);
                const selfNoGrad = this.detach();
                const otherNoGrad = other.detach();

                if (this.requiresGrad) Tensor.addGrad(this, outGrad.mm(otherNoGrad.t()));
                if (other.requiresGrad) Tensor.addGrad(other, selfNoGrad.t().mm(outGrad));
            };
        }

        return out;
    }

    // Batched 3D tensor matmul
    bmm(other: TensorValue | Tensor): Tensor {
        other = this.handleOther(other);

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
                            batchA[q * batchAStrides[0] + i * batchAStrides[1] + k * batchAStrides[2] + this.offset] *
                            batchB[q * batchBStrides[0] + k * batchBStrides[1] + j * batchBStrides[2] + other.offset];
                    }
                }
            }
        }

        const out = new Tensor(batchC, { shape: batchCShape, strides: batchCStrides, numel: batchCSize });

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
                const outGrad = (out.grad as Tensor);
                const selfNoGrad = this.detach();
                const otherNoGrad = other.detach();

                if (this.requiresGrad) Tensor.addGrad(this, outGrad.bmm(otherNoGrad.transpose(1, 2)));
                if (other.requiresGrad) Tensor.addGrad(other, selfNoGrad.transpose(1, 2).bmm(outGrad));
            };
        }

        return out;
    }

    // Convert right-side 1D tensor to a vector (nx1 tensor) to do matmul
    mv(other: TensorValue | Tensor): Tensor {
        other = this.handleOther(other);

        // Verify 2D shape
        if (this.shape.length !== 2 || other.shape.length !== 1) {
            throw new Error("Input is not a 2D and 1D tensor pair");
        }

        return this.mm(other.unsqueeze(1)).squeeze(1);
    }

    // General matrix multiplication with different shapes
    matmul(other: TensorValue | Tensor): Tensor {
        other = this.handleOther(other);

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
            (this.shape.length > 0 && other.shape.length >= 2) ||
            (this.shape.length >= 2 && other.shape.length > 0)
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
            // Base offset data
            const offsetShape = Tensor.broadcastShapes(selfOffsetShape, otherOffsetShape);
            const offsetSize = Tensor.shapeToSize(offsetShape);
            const offsetStrides = Tensor.getStrides(offsetShape);
            // Output shape, strides, size, value
            const outputShape = [...offsetShape, batchARows, batchBCols];
            const outputStrides = Tensor.getStrides(outputShape);
            const outputSize = Tensor.shapeToSize(outputShape);
            const outputValue = new Array(outputSize).fill(0);
            const outputOffsetStrides = outputStrides.slice(0, -2);

            // Loop through outer dims and do matmul on two outer-most dims
            for (let index = 0; index < offsetSize; index++) {
                const coords = Tensor.indexToCoords(index, offsetStrides);
                const offset = Tensor.coordsToIndex(coords, outputOffsetStrides);
                const selfOffset = Tensor.coordsToUnbroadcastedIndex(coords, selfOffsetShape, selfOffsetStrides);
                const otherOffset = Tensor.coordsToUnbroadcastedIndex(coords, otherOffsetShape, otherOffsetStrides);

                for (let i = 0; i < batchARows; i++) {
                    for (let j = 0; j < batchBCols; j++) {
                        for (let k = 0; k < batchACols; k++) {
                            const outputIdx = offset + i * outputStrides[lastDim - 1] + j * outputStrides[lastDim];
                            const selfIdx = selfOffset + i * selfStrides[lastDim - 1] + k * selfStrides[lastDim];
                            const otherIdx = otherOffset + k * otherStrides[lastDim - 1] + j * otherStrides[lastDim];

                            outputValue[outputIdx] += batchA[selfIdx + this.offset] * batchB[otherIdx + other.offset];
                        }
                    }
                }
            }

            const out = new Tensor(outputValue, { shape: outputShape, strides: outputStrides, numel: outputSize });

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
                    const outGrad = (out.grad as Tensor);
                    const selfNoGrad = self.detach();
                    const otherNoGrad = other.detach();

                    if (this.requiresGrad) Tensor.addGrad(this, outGrad.matmul(otherNoGrad.transpose(-2, -1)));
                    if (other.requiresGrad) Tensor.addGrad(other, selfNoGrad.transpose(-2, -1).matmul(outGrad));
                };
            }

            return out;
        }

        throw new Error(`Shapes [${this.shape}] and [${other.shape}] are not supported`);
    }

    // Dropout
    dropout(rate: number): Tensor {
        if (!Tensor.training || rate === 0) return this;

        const keepRate = 1 - rate;
        const uniform = Tensor.randLike(this);
        const mask = uniform.lt(keepRate);

        return this.mul(mask).div(keepRate);
    }

    // Get the upper triangular part with respect to main diagonal
    triu(diagonal=0): Tensor {
        if (this.shape.length < 2) {
            throw new Error("triu requires at least 2 dimensions");
        }

        const maskShape = this.shape.slice(-2);
        const maskStrides = Tensor.getStrides(maskShape);
        const maskSize = Tensor.shapeToSize(maskShape);
        const maskValue = new Array(maskSize).fill(1);

        const [rows, cols] = maskShape;

        for (let i = 0; i < rows; i++) {
            const maxJ = Math.min(cols, i + diagonal);
            for (let j = 0; j < maxJ; j++) {
                maskValue[i * maskStrides[0] + j * maskStrides[1]] = 0;
            }
        }

        const mask = new Tensor(maskValue, {
            shape: maskShape,
            strides: maskStrides,
            numel: maskSize,
            device: this.device
        });

        return this.mul(mask);
    }

    // Get the lower triangular part with respect to main diagonal
    tril(diagonal=0): Tensor {
        if (this.shape.length < 2) {
            throw new Error("triu requires at least 2 dimensions");
        }

        const maskShape = this.shape.slice(-2);
        const maskStrides = Tensor.getStrides(maskShape);
        const maskSize = Tensor.shapeToSize(maskShape);
        const maskValue = new Array(maskSize).fill(0);

        const [rows, cols] = maskShape;

        for (let i = 0; i < rows; i++) {
            const maxJ = Math.min(cols, i + diagonal + 1);
            for (let j = 0; j < maxJ; j++) {
                maskValue[i * maskStrides[0] + j * maskStrides[1]] = 1;
            }
        }

        const mask = new Tensor(maskValue, {
            shape: maskShape,
            strides: maskStrides,
            numel: maskSize,
            device: this.device
        });

        return this.mul(mask);
    }

    // Utility to create a new tensor filled with a number
    static full(shape: readonly number[], num: number, options: TensorOptions = {}): Tensor {
        if (shape.length === 0) return new Tensor(num, options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize).fill(num);

        return new Tensor(outputValue, { shape, numel: outputSize, ...options });
    }

    // Utility to create a new tensor with shape of another tensor, filled with a number
    static fullLike(tensor: Tensor, num: number, options: TensorOptions = {}): Tensor {
        if (typeof tensor.value === "number") return new Tensor(num, options);

        return new Tensor(new Array(tensor.numel).fill(num), {
            shape: tensor.shape,
            numel: tensor.numel,
            device: tensor.device,
            ...options
        });
    }

    // Utility to create a new tensor filled with 1
    static ones(shape?: readonly number[], options: TensorOptions = {}): Tensor {
        if (typeof shape === "undefined" || shape.length === 0) return new Tensor(1, options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize).fill(1);

        return new Tensor(outputValue, { shape, numel: outputSize, ...options });
    }

    // Utility to create a new tensor with shape of another tensor, filled with 1
    static onesLike(tensor: Tensor, options: TensorOptions = {}): Tensor {
        if (typeof tensor.value === "number") return new Tensor(1, options);

        return new Tensor(new Array(tensor.numel).fill(1), {
            shape: tensor.shape,
            numel: tensor.numel,
            device: tensor.device,
            ...options
        });
    }

    // Utility to create a new tensor filled with 0
    static zeros(shape?: readonly number[], options: TensorOptions = {}): Tensor {
        if (typeof shape === "undefined" || shape.length === 0) return new Tensor(0, options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize).fill(0);

        return new Tensor(outputValue, { shape, numel: outputSize, ...options });
    }

    // Utility to create a new tensor with shape of another tensor, filled with 0
    static zerosLike(tensor: Tensor, options: TensorOptions = {}): Tensor {
        if (typeof tensor.value === "number") return new Tensor(0, options);

        return new Tensor(new Array(tensor.numel).fill(0), {
            shape: tensor.shape,
            numel: tensor.numel,
            device: tensor.device,
            ...options
        });
    }

    // Utility to create a new tensor filled with a random number with uniform distribution from 0 to 1
    static rand(shape?: readonly number[], options: TensorOptions = {}): Tensor {
        if (typeof shape === "undefined" || shape.length === 0) return new Tensor(randUniform(), options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randUniform();
        }

        return new Tensor(outputValue, { shape, numel: outputSize, ...options });
    }

    // Utility to create a new tensor with shape of another tensor, filled with a random number with uniform distribution from 0 to 1
    static randLike(tensor: Tensor, options: TensorOptions = {}): Tensor {
        if (typeof tensor.value === "number") return new Tensor(randUniform(), options);

        const outputValue = new Array(tensor.numel);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randUniform();
        }

        return new Tensor(outputValue, {
            shape: tensor.shape,
            numel: tensor.numel,
            device: tensor.device,
            ...options
        });
    }

    // Utility to create a new tensor filled with a random number with normal distribution of mean=0 and stddev=1
    static randn(shape?: readonly number[], options: TensorOptions = {}): Tensor {
        if (typeof shape === "undefined" || shape.length === 0) return new Tensor(randNormal(), options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randNormal();
        }

        return new Tensor(outputValue, { shape, numel: outputSize, ...options });
    }

    // Utility to create a new tensor with shape of another tensor, filled with a random number with normal distribution of mean=0 and stddev=1
    static randnLike(tensor: Tensor, options: TensorOptions = {}): Tensor {
        if (typeof tensor.value === "number") return new Tensor(randNormal(), options);

        const outputValue = new Array(tensor.numel);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randNormal();
        }

        return new Tensor(outputValue, {
            shape: tensor.shape,
            numel: tensor.numel,
            device: tensor.device,
            ...options
        });
    }

    // Utility to create a new tensor filled with a random integer between low and high
    static randint(shape: readonly number[], low: number, high: number, options: TensorOptions = {}): Tensor {
        if (shape.length === 0) return new Tensor(randInt(low, high), options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randInt(low, high);
        }

        return new Tensor(outputValue, { shape, numel: outputSize, ...options });
    }

    // Utility to create a new tensor with shape of another tensor, filled with a random integer between low and high
    static randintLike(tensor: Tensor, low: number, high: number, options: TensorOptions = {}): Tensor {
        if (typeof tensor.value === "number") return new Tensor(randInt(low, high), options);

        const outputValue = new Array(tensor.numel);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randInt(low, high);
        }

        return new Tensor(outputValue, {
            shape: tensor.shape,
            numel: tensor.numel,
            device: tensor.device,
            ...options
        });
    }

    // Utility to create a new tensor filled with integers from 0 to n, randomly shuffled
    static randperm(n: number, options: TensorOptions = {}): Tensor {
        const outputValue = new Array(n);

        for (let i = 0; i < n; i++) {
            outputValue[i] = i;
        }

        fyShuffle(outputValue);

        return new Tensor(outputValue, { shape: [n], numel: n, ...options });
    }

    // Utility to create a new tensor filled with a random number with normal distribution of custom mean and stddev
    static normal(shape: number[], mean: number, stdDev: number, options: TensorOptions = {}): Tensor {
        if (shape.length === 0) return new Tensor(randNormal(mean, stdDev), options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randNormal(mean, stdDev);
        }

        return new Tensor(outputValue, { shape, numel: outputSize, ...options })
    }

    // Utility to create a new tensor filled with a random number with uniform distribution from low to high
    static uniform(shape: number[], low: number, high: number, options: TensorOptions = {}): Tensor {
        if (shape.length === 0) return new Tensor(randUniform(low, high), options);

        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = randUniform(low, high);
        }

        return new Tensor(outputValue, { shape, numel: outputSize, ...options })
    }

    // Utility to create an 1D tensor from a range incrementing with "step"
    static arange(start: number, stop?: number, step = 1, options: TensorOptions = {}): Tensor {
        if (typeof stop === "undefined") {
            stop = start;
            start = 0;
        }

        const outputSize = Math.ceil((stop - start) / step);
        const outputShape = [outputSize];
        const outputValue = new Array(outputSize);

        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = start + step * index;
        }

        return new Tensor(outputValue, { shape: outputShape, numel: outputSize, ...options })
    }

    // Utility to create an 1D tensor from a range evenly spaced out with a given amount of steps
    static linspace(start: number, stop: number, steps: number, options: TensorOptions = {}): Tensor {
        if (steps <= 0) throw new Error("Steps must be positive");
        if (steps === 1) {
            return new Tensor([start], { shape: [1], numel: 1, ...options });
        }

        const step = (stop - start) / (steps - 1);
        const outputValue = new Array(steps);

        for (let index = 0; index < steps; index++) {
            outputValue[index] = start + step * index;
        }
        // Ensure we hit the endpoint exactly (avoids floating point errors)
        outputValue[steps - 1] = stop;

        return new Tensor(outputValue, { shape: [steps], numel: steps, ...options });
    }

    // Utility to create a 2D tensor with its main diagonal filled with 1s and others with 0s
    static eye(n: number, m: number = n, options: TensorOptions = {}): Tensor {
        const outputSize = n * m;
        const outputShape = [n, m];
        const outputStrides = Tensor.getStrides(outputShape);
        const outputValue = new Array(outputSize).fill(0);

        for (let i = 0; i < Math.min(n, m); i++) {
            outputValue[i * outputStrides[0] + i * outputStrides[1]] = 1;
        }

        return new Tensor(outputValue, { shape: outputShape, strides: outputStrides, numel: outputSize, ...options })
    }

    // Reverse-mode autodiff call
    backward(options: { zeroGrad?: boolean } = {}) {
        // Init
        const zeroGrad = options.zeroGrad ?? true;

        // Build topological order
        const topo: Tensor[] = [];
        const visited: Set<Tensor> = new Set();

        function build(node: Tensor) {
            // Only collects unvisited node and node that requires gradient
            if (!visited.has(node) && node.requiresGrad) {
                visited.add(node);

                // Reset grad to zeros if specified
                if (zeroGrad) {
                    node.grad = Tensor.zerosLike(node);
                }

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

        return buildNested(this.value, this.shape, this.strides, this.offset);
    }

    // Returns a view of the tensor with gradient turned on/off and detaches from autograd
    withGrad(requiresGrad: boolean): Tensor {
        return new Tensor(this.value, {
            shape: this.shape,
            strides: this.strides,
            offset: this.offset,
            numel: this.numel,
            device: this.device,
            requiresGrad
        })
    }

    // Returns a view of the tensor with gradient turned off and detaches from autograd
    detach(): Tensor {
        return new Tensor(this.value, {
            shape: this.shape,
            strides: this.strides,
            offset: this.offset,
            numel: this.numel,
            device: this.device,
            requiresGrad: false
        })
    }

    // Returns a copy of the tensor (with new data allocation) and detaches from autograd
    clone(): Tensor {
        return new Tensor(typeof this.value === "number" ? this.value : [...this.value], {
            shape: this.shape,
            strides: this.strides,
            offset: this.offset,
            numel: this.numel,
            requiresGrad: this.requiresGrad
        })
    }

    // Returns this tensor with value replaced with the value of another tensor
    replace(other: Tensor | TensorValue, allowShapeMismatch: boolean = false): Tensor {
        other = this.handleOther(other);

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
        if (device === "cpu") return this;

        const backend = Tensor.backends.get(device);

        if (backend && backend.transfer) {
            return backend.transfer(this);
        }

        throw new Error(`No device found to transfer tensor to or a handler is not implemented for device.`);
    }

    // Op to transfer tensor to another device in-place
    to_(device: string): Tensor {
        if (device === "cpu") return this;

        const backend = Tensor.backends.get(this.device);

        if (backend && backend.create) {
            backend.create(this);
            return this;
        }

        throw new Error(`No device found to transfer tensor to or a handler is not implemented for device.`);
    }
}

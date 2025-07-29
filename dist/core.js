"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Tensor = void 0;
const utils_1 = require("./utils");
class Tensor {
    value;
    shape;
    strides;
    grad;
    requiresGrad;
    gradFn;
    children;
    constructor(value, options = {}) {
        this.value = Tensor.flatten(value);
        this.shape = options.shape || Tensor.getShape(value);
        this.strides = options.strides || Tensor.getStrides(this.shape);
        this.grad = options.grad;
        this.requiresGrad = options.requiresGrad ?? false;
        this.gradFn = options.gradFn || (() => { });
        this.children = options.children || [];
    }
    // Utility to flatten an nD array to be 1D
    static flatten(tensor) {
        // Handle scalar tensors
        if (typeof tensor === "number")
            return tensor;
        // If value is already 1D, we just need to return the value ('s reference)
        if (typeof tensor[0] === "number")
            return tensor;
        // Or else recursively traverse through the nD array to flatten
        const result = [];
        function traverse(arr) {
            if (typeof arr === "number") {
                result.push(arr);
            }
            else if (Array.isArray(arr)) {
                arr.forEach(traverse);
            }
        }
        traverse(tensor);
        return result;
    }
    // Utility to get shape from tensor *value*
    static getShape(tensor) {
        const shape = [];
        let subA = tensor;
        while (Array.isArray(subA)) {
            shape.push(subA.length);
            subA = subA[0];
        }
        return shape;
    }
    // Utility to get strides from shape
    static getStrides(shape) {
        const strides = new Array(shape.length);
        strides[strides.length - 1] = 1;
        for (let i = strides.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }
    // Left-pad shape and strides for two shape to be of same length
    static padShape(stridesA, stridesB, shapeA, shapeB) {
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
    static broadcastShapes(shapeA, shapeB) {
        const newShape = new Array(shapeA.length);
        for (let index = 0; index < shapeA.length; index++) {
            if (shapeA[index] === 1) {
                newShape[index] = shapeB[index];
            }
            else if (shapeB[index] === 1) {
                newShape[index] = shapeA[index];
            }
            else if (shapeA[index] === shapeB[index]) {
                newShape[index] = shapeA[index];
            }
            else {
                throw new Error(`Cannot broadcast shapes: ${shapeA} and ${shapeB}`);
            }
        }
        return newShape;
    }
    // Utility to convert flat index to array of coordinates
    static indexToCoords(index, strides) {
        const coords = new Array(strides.length);
        let remaining = index;
        for (let dim = 0; dim < strides.length; dim++) {
            coords[dim] = Math.floor(remaining / strides[dim]);
            remaining %= strides[dim];
        }
        return coords;
    }
    // Utility to convert array of coordinates to *unbroadcasted* flat index 
    static coordsToUnbroadcastedIndex(coords, shape, strides) {
        let index = 0;
        for (let i = 0; i < coords.length; i++) {
            // Handle broadcasting
            const actualCoord = shape[i] === 1 ? 0 : coords[i];
            index += actualCoord * strides[i];
        }
        return index;
    }
    // Utility to convert array of coordinates to flat index 
    static coordsToIndex(coords, strides) {
        let index = 0;
        for (let i = 0; i < coords.length; i++) {
            index += coords[i] * strides[i];
        }
        return index;
    }
    // Utility to convert shape into 1D value array size
    static shapeToSize(shape) {
        let prod = 1;
        for (let i = 0; i < shape.length; i++) {
            prod *= shape[i];
        }
        return prod;
    }
    ;
    // Utility for binary (two operators involved) element-wise ops
    static elementWiseAB(tA, tB, op) {
        if (typeof tA.value === "number" && typeof tB.value === "number") {
            return new Tensor(op(tA.value, tB.value));
        }
        if (typeof tA.value === "number") {
            return Tensor.elementWiseSelf(tB, (a) => op(a, tA.value));
        }
        if (typeof tB.value === "number") {
            return Tensor.elementWiseSelf(tA, (a) => op(a, tB.value));
        }
        // Pad + broadcast shape
        const [paddedAStrides, paddedBStrides, paddedAShape, paddedBShape] = Tensor.padShape(tA.strides, tB.strides, tA.shape, tB.shape);
        const outputShape = Tensor.broadcastShapes(paddedAShape, paddedBShape);
        // Get other output info
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize);
        for (let i = 0; i < outputSize; i++) {
            // Get coordinates from 1D index
            const coordsOutput = Tensor.indexToCoords(i, outputStrides);
            // Convert the coordinates to 1D index of flattened A with respect to A's shape
            const indexA = Tensor.coordsToUnbroadcastedIndex(coordsOutput, paddedAShape, paddedAStrides);
            // Convert the coordinates to 1D index of flattened B with respect to B's shape
            const indexB = Tensor.coordsToUnbroadcastedIndex(coordsOutput, paddedBShape, paddedBStrides);
            // Calculate with op
            outputValue[i] = op(tA.value[indexA], tB.value[indexB]);
        }
        return new Tensor(outputValue, {
            shape: outputShape,
            strides: outputStrides
        });
    }
    // Utility for self-inflicting element-wise ops
    static elementWiseSelf(tA, op) {
        if (typeof tA.value === "number")
            return new Tensor(op(tA.value));
        const newValue = new Array(tA.value.length);
        for (let index = 0; index < tA.value.length; index++) {
            newValue[index] = op(tA.value[index]);
        }
        return new Tensor(newValue, { shape: tA.shape, strides: tA.strides });
    }
    // Utility to do element-wise operation and build a dag node with another tensor
    elementWiseABDAG(other, op, thisGrad = () => new Tensor(0), otherGrad = () => new Tensor(0)) {
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
                const outGrad = out.grad.withGrad(false);
                const selfNoGrad = this.withGrad(false);
                const otherNoGrad = other.withGrad(false);
                if (this.requiresGrad)
                    Tensor.addGrad(this, thisGrad(selfNoGrad, otherNoGrad, outGrad));
                if (other.requiresGrad)
                    Tensor.addGrad(other, otherGrad(selfNoGrad, otherNoGrad, outGrad));
            };
        }
        return out;
    }
    // Utility to do self-inflicting element-wise operation and build a dag node
    elementWiseSelfDAG(op, thisGrad = () => new Tensor(0)) {
        const out = Tensor.elementWiseSelf(this, op);
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
        }
        if (out.requiresGrad) {
            out.gradFn = () => {
                // Disable gradient collecting of gradients themselves
                const outGrad = out.grad.withGrad(false);
                const selfNoGrad = this.withGrad(false);
                if (this.requiresGrad)
                    Tensor.addGrad(this, thisGrad(selfNoGrad, outGrad));
            };
        }
        return out;
    }
    // Utility to force an input value to be a tensor
    static forceTensor(value) {
        if (value instanceof Tensor)
            return value;
        return new Tensor(value);
    }
    // Utility to add to gradient of tensor
    static addGrad(tensor, accumGrad) {
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
        }
        else {
            tensor.grad = tensor.grad.add(squeezedGrad);
        }
    }
    // Tensor squeeze
    squeeze(dims) {
        if (typeof this.value === "number")
            return this;
        if (typeof dims === "number") {
            dims = [dims];
        }
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
        const outShape = [], outStrides = [];
        for (let index = 0; index < this.shape.length; index++) {
            const dim = this.shape[index];
            const stride = this.strides[index];
            if (dims.includes(index)) {
                if (dim !== 1)
                    throw new Error(`Can not squeeze dim with size ${dim}`);
            }
            else {
                outShape.push(dim);
                outStrides.push(stride);
            }
        }
        const outValue = outShape.length === 0 ? this.value[0] : this.value;
        const out = new Tensor(outValue, {
            shape: outShape,
            strides: outStrides
        });
        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                let restoredGrad = out.grad.withGrad(false);
                for (let i = dims.length - 1; i >= 0; i--) {
                    restoredGrad = restoredGrad.unsqueeze(dims[i]);
                }
                Tensor.addGrad(this, restoredGrad);
            };
        }
        return out;
    }
    // Tensor unsqueeze - adds dimension of size 1 at specified position
    unsqueeze(dim) {
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
        }
        else {
            // Inserting before dim: use current stride * current shape
            newDimStride = this.strides[dim] * this.shape[dim];
        }
        newStrides.splice(dim, 0, newDimStride);
        const out = new Tensor(thisValue, { shape: newShape, strides: newStrides });
        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, out.grad.withGrad(false).squeeze(dim));
            };
        }
        return out;
    }
    // Tensor sum reduction
    sum(dims, keepDims = false) {
        if (typeof this.value === "number")
            return this;
        if (typeof dims === "number") {
            dims = [dims];
        }
        if (typeof dims === "undefined") {
            dims = Array.from({ length: this.shape.length }, (_, index) => index);
        }
        // Dims that are reduced now have size-1
        const outputShape = this.shape.map((dim, i) => dims.includes(i) ? 1 : dim);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize).fill(0);
        const originalSize = Tensor.shapeToSize(this.shape);
        // Gradient data
        let gradShape, gradStrides, gradValue = [];
        // Allocate gradient data only when needed
        if (this.requiresGrad) {
            gradShape = this.shape;
            gradStrides = this.strides;
            gradValue = new Array(originalSize).fill(0);
        }
        // Calculate new value after sum
        for (let index = 0; index < originalSize; index++) {
            const coords = Tensor.indexToCoords(index, this.strides);
            // Force 0 on reduced axes to collapse into size-1 dims
            const outCoords = coords.map((val, i) => dims.includes(i) ? 0 : val);
            // Convert output coordinates to flat index
            const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
            // Accumulate, outFlatIndex should match multiple realFlatIndexes
            const realFlatIndex = Tensor.coordsToIndex(coords, this.strides);
            // Add into sum
            outputValue[outFlatIndex] += this.value[realFlatIndex];
            // Mark for gradient if needed
            if (this.requiresGrad) {
                gradValue[realFlatIndex] = 1;
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
                const localGrad = new Tensor(gradValue, { shape: gradShape, strides: gradStrides });
                Tensor.addGrad(this, out.grad.withGrad(false).mul(localGrad));
            };
        }
        return keepDims ? out : out.squeeze(dims);
    }
    // Tensor product reduction
    prod(dims, keepDims = false) {
        if (typeof this.value === "number")
            return this;
        if (typeof dims === "number") {
            dims = [dims];
        }
        if (typeof dims === "undefined") {
            dims = Array.from({ length: this.shape.length }, (_, index) => index);
        }
        // Dims that are reduced now have size-1
        const outputShape = this.shape.map((dim, i) => dims.includes(i) ? 1 : dim);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize).fill(1);
        const originalSize = Tensor.shapeToSize(this.shape);
        // Calculate new value after multiplying
        for (let index = 0; index < originalSize; index++) {
            const coords = Tensor.indexToCoords(index, this.strides);
            // Force 0 on reduced axes to collapse into size-1 dims
            const outCoords = coords.map((val, i) => dims.includes(i) ? 0 : val);
            // Convert output coordinates to flat index
            const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
            // Accumulate, outFlatIndex should match multiple realFlatIndexes
            const realFlatIndex = Tensor.coordsToIndex(coords, this.strides);
            // Multiply into product
            outputValue[outFlatIndex] *= this.value[realFlatIndex];
        }
        const out = new Tensor(outputValue, {
            shape: outputShape,
            strides: outputStrides
        });
        // Set up gradient if needed
        if (this.requiresGrad) {
            const gradShape = this.shape, gradStrides = this.strides, gradValue = new Array(originalSize).fill(0);
            for (let index = 0; index < originalSize; index++) {
                const coords = Tensor.indexToCoords(index, this.strides);
                // Force 0 on reduced axes to collapse into size-1 dims
                const outCoords = coords.map((val, i) => dims.includes(i) ? 0 : val);
                // Convert output coordinates to flat index
                const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
                // Accumulate, outFlatIndex should match multiple realFlatIndexes
                const realFlatIndex = Tensor.coordsToIndex(coords, this.strides);
                // Grad is the product of other elements of the same axis, which is product of all els divided by the current value
                gradValue[realFlatIndex] = outputValue[outFlatIndex] / this.value[realFlatIndex];
            }
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                const localGrad = new Tensor(gradValue, { shape: gradShape, strides: gradStrides });
                Tensor.addGrad(this, out.grad.withGrad(false).mul(localGrad));
            };
        }
        return keepDims ? out : out.squeeze(dims);
    }
    // Tensor mean reduction
    mean(dims, keepDims = false) {
        if (typeof this.value === "number")
            return this;
        if (typeof dims === "number") {
            dims = [dims];
        }
        if (typeof dims === "undefined") {
            dims = Array.from({ length: this.shape.length }, (_, index) => index);
        }
        // Dims that are reduced now have size-1
        const outputShape = this.shape.map((dim, i) => dims.includes(i) ? 1 : dim);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize).fill(0);
        const outputFeeders = new Array(outputSize).fill(0);
        const originalSize = Tensor.shapeToSize(this.shape);
        // Calculate sums and how many elements contribute to specific positions
        for (let index = 0; index < originalSize; index++) {
            const coords = Tensor.indexToCoords(index, this.strides);
            // Force 0 on reduced axes to collapse into size-1 dims
            const outCoords = coords.map((val, i) => dims.includes(i) ? 0 : val);
            // Convert output coordinates to flat index
            const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
            // Accumulate, outFlatIndex should match multiple realFlatIndexes
            const realFlatIndex = Tensor.coordsToIndex(coords, this.strides);
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
            const gradShape = this.shape, gradStrides = this.strides, gradValue = new Array(originalSize).fill(0);
            // Calculate grad by assigning 1 divided by the number of contributors to the position
            for (let index = 0; index < originalSize; index++) {
                const coords = Tensor.indexToCoords(index, this.strides);
                // Force 0 on reduced axes to collapse into size-1 dims
                const outCoords = coords.map((val, i) => dims.includes(i) ? 0 : val);
                // Convert output coordinates to flat index
                const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
                // Accumulate, outFlatIndex should match multiple realFlatIndexes
                const realFlatIndex = Tensor.coordsToIndex(coords, this.strides);
                // Mean = 1/n * (el1 + el2 + ... + eln) so grad = 1/n
                gradValue[realFlatIndex] = 1 / outputFeeders[outFlatIndex];
            }
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                const localGrad = new Tensor(gradValue, { shape: gradShape, strides: gradStrides });
                Tensor.addGrad(this, out.grad.withGrad(false).mul(localGrad));
            };
        }
        return keepDims ? out : out.squeeze(dims);
    }
    // Tensor maximum reduction
    max(dims, keepDims = false) {
        if (typeof this.value === "number")
            return this;
        if (typeof dims === "number") {
            dims = [dims];
        }
        if (typeof dims === "undefined") {
            dims = Array.from({ length: this.shape.length }, (_, index) => index);
        }
        // Dims that are reduced now have size-1
        const outputShape = this.shape.map((dim, i) => dims.includes(i) ? 1 : dim);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize).fill(-Infinity);
        const originalSize = Tensor.shapeToSize(this.shape);
        // Calculate maximum values of axes
        for (let index = 0; index < originalSize; index++) {
            const coords = Tensor.indexToCoords(index, this.strides);
            // Force 0 on reduced axes to collapse into size-1 dims
            const outCoords = coords.map((val, i) => dims.includes(i) ? 0 : val);
            // Convert output coordinates to flat index
            const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
            // Accumulate, outFlatIndex should match multiple realFlatIndexes
            const realFlatIndex = Tensor.coordsToIndex(coords, this.strides);
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
            const gradShape = this.shape, gradStrides = this.strides, gradValue = new Array(originalSize).fill(0);
            for (let index = 0; index < originalSize; index++) {
                const coords = Tensor.indexToCoords(index, this.strides);
                // Force 0 on reduced axes to collapse into size-1 dims
                const outCoords = coords.map((val, i) => dims.includes(i) ? 0 : val);
                // Convert output coordinates to flat index
                const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
                // Accumulate, outFlatIndex should match multiple realFlatIndexes
                const realFlatIndex = Tensor.coordsToIndex(coords, this.strides);
                // Calculate grad by checking if a positon holds a value equal to the max value
                gradValue[realFlatIndex] = outputValue[outFlatIndex] === this.value[realFlatIndex] ? 1 : 0;
            }
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                const localGrad = new Tensor(gradValue, { shape: gradShape, strides: gradStrides });
                Tensor.addGrad(this, out.grad.withGrad(false).mul(localGrad));
            };
        }
        return keepDims ? out : out.squeeze(dims);
    }
    // Tensor minimum reduction
    min(dims, keepDims = false) {
        if (typeof this.value === "number")
            return this;
        if (typeof dims === "number") {
            dims = [dims];
        }
        if (typeof dims === "undefined") {
            dims = Array.from({ length: this.shape.length }, (_, index) => index);
        }
        // Dims that are reduced now have size-1
        const outputShape = this.shape.map((dim, i) => dims.includes(i) ? 1 : dim);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new Array(outputSize).fill(Infinity);
        const originalSize = Tensor.shapeToSize(this.shape);
        // Calculate minimum values of axes
        for (let index = 0; index < originalSize; index++) {
            const coords = Tensor.indexToCoords(index, this.strides);
            // Force 0 on reduced axes to collapse into size-1 dims
            const outCoords = coords.map((val, i) => dims.includes(i) ? 0 : val);
            // Convert output coordinates to flat index
            const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
            // Accumulate, outFlatIndex should match multiple realFlatIndexes
            const realFlatIndex = Tensor.coordsToIndex(coords, this.strides);
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
            const gradShape = this.shape, gradStrides = this.strides, gradValue = new Array(originalSize).fill(0);
            for (let index = 0; index < originalSize; index++) {
                const coords = Tensor.indexToCoords(index, this.strides);
                // Force 0 on reduced axes to collapse into size-1 dims
                const outCoords = coords.map((val, i) => dims.includes(i) ? 0 : val);
                // Convert output coordinates to flat index
                const outFlatIndex = Tensor.coordsToIndex(outCoords, outputStrides);
                // Accumulate, outFlatIndex should match multiple realFlatIndexes
                const realFlatIndex = Tensor.coordsToIndex(coords, this.strides);
                // Calculate grad by checking if a positon holds a value equal to the min value
                gradValue[realFlatIndex] = outputValue[outFlatIndex] === this.value[realFlatIndex] ? 1 : 0;
            }
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                const localGrad = new Tensor(gradValue, { shape: gradShape, strides: gradStrides });
                Tensor.addGrad(this, out.grad.withGrad(false).mul(localGrad));
            };
        }
        return keepDims ? out : out.squeeze(dims);
    }
    // Tensor element-wise addition
    add(other) {
        return this.elementWiseABDAG(other, (a, b) => a + b, (self, other, outGrad) => outGrad, (self, other, outGrad) => outGrad);
    }
    // Tensor element-wise subtraction
    sub(other) {
        return this.elementWiseABDAG(other, (a, b) => a - b, (self, other, outGrad) => outGrad, (self, other, outGrad) => outGrad.neg());
    }
    subtract = this.sub;
    // Tensor element-wise multiplication
    mul(other) {
        return this.elementWiseABDAG(other, (a, b) => a * b, (self, other, outGrad) => outGrad.mul(other), (self, other, outGrad) => outGrad.mul(self));
    }
    multiply = this.mul;
    // Tensor element-wise power
    pow(other) {
        return this.elementWiseABDAG(other, (a, b) => a ** b, (self, other, outGrad) => outGrad.mul(other.mul(self.pow(other.sub(1)))), (self, other, outGrad) => outGrad.mul(self.pow(other).mul(self.log())));
    }
    // Tensor element-wise division
    div(other) {
        return this.elementWiseABDAG(other, (a, b) => a / b, (self, other, outGrad) => outGrad.div(other), (self, other, outGrad) => outGrad.mul(self.neg().div(other.square())));
    }
    divide = this.div;
    // Tensor element-wise modulo
    remainder(other) {
        return this.elementWiseABDAG(other, (a, b) => a % b);
    }
    // Tensor element-wise greater or equal comparison
    ge(other) {
        return this.elementWiseABDAG(other, (a, b) => a >= b ? 1 : 0);
    }
    greaterEqual = this.ge;
    // Tensor element-wise less or equal comparison
    le(other) {
        return this.elementWiseABDAG(other, (a, b) => a <= b ? 1 : 0);
    }
    lessEqual = this.le;
    // Tensor element-wise greater-than comparison
    gt(other) {
        return this.elementWiseABDAG(other, (a, b) => a > b ? 1 : 0);
    }
    greater = this.gt;
    // Tensor element-wise less-than comparison
    lt(other) {
        return this.elementWiseABDAG(other, (a, b) => a < b ? 1 : 0);
    }
    less = this.lt;
    // Tensor element-wise equality comparison
    eq(other) {
        return this.elementWiseABDAG(other, (a, b) => a === b ? 1 : 0);
    }
    equal = this.eq;
    // Tensor element-wise not equality comparison
    ne(other) {
        return this.elementWiseABDAG(other, (a, b) => a !== b ? 1 : 0);
    }
    notEqual = this.ne;
    // Tensor element-wise logical and
    logicalAnd(other) {
        return this.elementWiseABDAG(other, (a, b) => a === 1 && b === 1 ? 1 : 0);
    }
    // Tensor element-wise logical or
    logicalOr(other) {
        return this.elementWiseABDAG(other, (a, b) => a === 1 || b === 1 ? 1 : 0);
    }
    // Tensor element-wise logical xor
    logicalXor(other) {
        return this.elementWiseABDAG(other, (a, b) => (a === 1 || b === 1) && a !== b ? 1 : 0);
    }
    // Tensor element-wise logical not
    logicalNot() {
        return this.elementWiseSelfDAG((a) => a === 1 ? 0 : 1);
    }
    // Tensor element-wise bitwise and
    bitwiseAnd(other) {
        return this.elementWiseABDAG(other, (a, b) => a & b);
    }
    // Tensor element-wise bitwise or
    bitwiseOr(other) {
        return this.elementWiseABDAG(other, (a, b) => a | b);
    }
    // Tensor element-wise bitwise xor
    bitwiseXor(other) {
        return this.elementWiseABDAG(other, (a, b) => a ^ b);
    }
    // Tensor element-wise bitwise not
    bitwiseNot() {
        return this.elementWiseSelfDAG((a) => ~a);
    }
    // Tensor element-wise left shift
    bitwiseLeftShift(other) {
        return this.elementWiseABDAG(other, (a, b) => a << b);
    }
    // Tensor element-wise right shift
    bitwiseRightShift(other) {
        return this.elementWiseABDAG(other, (a, b) => a >> b);
    }
    // Tensor element-wise negation
    neg() {
        return this.elementWiseSelfDAG((a) => -a, (self, outGrad) => outGrad.mul(-1));
    }
    negative = this.neg;
    // Tensor element-wise reciprocal
    reciprocal() {
        return this.elementWiseSelfDAG((a) => 1 / a, (self, outGrad) => outGrad.mul(self.pow(-2).neg()));
    }
    // Tensor element-wise square
    square() {
        return this.elementWiseSelfDAG((a) => a * a, (self, outGrad) => outGrad.mul(self.mul(2)));
    }
    // Tensor element-wise absolute
    abs() {
        return this.elementWiseSelfDAG((a) => Math.abs(a), (self, outGrad) => outGrad.mul(self.sign()));
    }
    absolute = this.abs;
    // Tensor element-wise sign function
    sign() {
        return this.elementWiseSelfDAG((a) => Math.sign(a));
    }
    // Tensor element-wise sin
    sin() {
        return this.elementWiseSelfDAG((a) => Math.sin(a), (self, outGrad) => outGrad.mul(self.cos()));
    }
    // Tensor element-wise cos
    cos() {
        return this.elementWiseSelfDAG((a) => Math.cos(a), (self, outGrad) => outGrad.mul(self.sin().neg()));
    }
    // Tensor element-wise tan
    tan() {
        return this.elementWiseSelfDAG((a) => Math.tan(a), (self, outGrad) => outGrad.mul(self.tan().square().add(1)));
    }
    // Tensor element-wise asin
    asin() {
        return this.elementWiseSelfDAG((a) => Math.asin(a), (self, outGrad) => outGrad.div(self.square().neg().add(1).sqrt()));
    }
    arcsin = this.asin;
    // Tensor element-wise acos
    acos() {
        return this.elementWiseSelfDAG((a) => Math.acos(a), (self, outGrad) => outGrad.div(self.square().neg().add(1).sqrt()).neg());
    }
    arccos = this.acos;
    // Tensor element-wise atan
    atan() {
        return this.elementWiseSelfDAG((a) => Math.atan(a), (self, outGrad) => outGrad.div(self.square().add(1)));
    }
    arctan = this.atan;
    // Tensor element-wise atan2
    atan2(other) {
        return this.elementWiseABDAG(other, (a, b) => Math.atan2(a, b), (self, other, outGrad) => outGrad.mul(other.div(self.square().add(other.square()))), (self, other, outGrad) => outGrad.mul(self.neg().div(self.square().add(other.square()))));
    }
    arctan2 = this.atan2;
    // Tensor element-wise sinh
    sinh() {
        return this.elementWiseSelfDAG((a) => Math.sinh(a), (self, outGrad) => outGrad.mul(self.cosh()));
    }
    // Tensor element-wise cosh
    cosh() {
        return this.elementWiseSelfDAG((a) => Math.cosh(a), (self, outGrad) => outGrad.mul(self.sinh()));
    }
    // Tensor element-wise asinh
    asinh() {
        return this.elementWiseSelfDAG((a) => Math.asinh(a), (self, outGrad) => outGrad.div(self.square().add(1).sqrt()));
    }
    arcsinh = this.asinh;
    // Tensor element-wise acosh
    acosh() {
        return this.elementWiseSelfDAG((a) => Math.acosh(a), (self, outGrad) => outGrad.div(self.add(1).sqrt().mul(self.sub(1).sqrt())));
    }
    arccosh = this.acosh;
    // Tensor element-wise atanh
    atanh() {
        return this.elementWiseSelfDAG((a) => Math.atanh(a), (self, outGrad) => outGrad.div(self.square().neg().add(1)));
    }
    arctanh = this.atanh;
    // Tensor element-wise degree to radian
    deg2rad() {
        return this.elementWiseSelfDAG((a) => a * (Math.PI / 180), (self, outGrad) => outGrad.mul(Math.PI / 180));
    }
    // Tensor element-wise radian to degree
    rad2deg() {
        return this.elementWiseSelfDAG((a) => a / (Math.PI / 180), (self, outGrad) => outGrad.div(Math.PI / 180));
    }
    // Tensor element-wise square root
    sqrt() {
        return this.elementWiseSelfDAG((a) => Math.sqrt(a), (self, outGrad) => outGrad.div(self.sqrt().mul(2)));
    }
    // Tensor element-wise reciprocal of square root
    rsqrt() {
        return this.elementWiseSelfDAG((a) => 1 / Math.sqrt(a), (self, outGrad) => outGrad.mul(self.pow(-1.5).mul(-0.5)));
    }
    // Tensor element-wise e^x
    exp() {
        return this.elementWiseSelfDAG((a) => Math.exp(a), (self, outGrad) => outGrad.mul(self.exp()));
    }
    // Tensor element-wise 2^x
    exp2() {
        return this.elementWiseSelfDAG((a) => 2 ** a, (self, outGrad) => outGrad.mul(self.exp2().mul(Math.log(2))));
    }
    // Tensor element-wise e^x - 1
    expm1() {
        return this.elementWiseSelfDAG((a) => Math.expm1(a), (self, outGrad) => outGrad.mul(self.exp()));
    }
    // Tensor element-wise natural log
    log() {
        return this.elementWiseSelfDAG((a) => Math.log(a), (self, outGrad) => outGrad.div(self));
    }
    // Tensor element-wise log2
    log2() {
        return this.elementWiseSelfDAG((a) => Math.log2(a), (self, outGrad) => outGrad.div(self.mul(Math.log(2))));
    }
    // Tensor element-wise log10
    log10() {
        return this.elementWiseSelfDAG((a) => Math.log10(a), (self, outGrad) => outGrad.div(self.mul(Math.log(10))));
    }
    // Tensor element-wise log(1+x)
    log1p() {
        return this.elementWiseSelfDAG((a) => Math.log1p(a), (self, outGrad) => outGrad.div(self.add(1)));
    }
    // Tensor element-wise relu
    relu() {
        return this.elementWiseSelfDAG((a) => Math.max(a, 0), (self, outGrad) => outGrad.mul(self.gt(0)));
    }
    // Tensor element-wise sigmoid
    sigmoid() {
        return this.elementWiseSelfDAG((a) => 1 / (1 + Math.exp(-a)), (self, outGrad) => {
            const sig = self.sigmoid();
            return outGrad.mul(sig).mul(sig.neg().add(1));
        });
    }
    // Tensor element-wise tanh
    tanh() {
        return this.elementWiseSelfDAG((a) => Math.tanh(a), (self, outGrad) => outGrad.mul(self.tanh().square().neg().add(1)));
    }
    // Tensor element-wise softplus
    softplus() {
        return this.elementWiseSelfDAG((a) => Math.log1p(Math.exp(a)), (self, outGrad) => outGrad.mul(self.sigmoid()));
    }
    // Tensor element-wise softsign
    softsign() {
        return this.elementWiseSelfDAG((a) => a / (1 + Math.abs(a)), (self, outGrad) => outGrad.div(self.abs().add(1).square()));
    }
    // Tensor element-wise silu (swish)
    silu() {
        return this.elementWiseSelfDAG((a) => a / (1 + Math.exp(-a)), (self, outGrad) => {
            const sig = self.sigmoid();
            return outGrad.mul(sig.add(self.mul(sig).mul(sig.neg().add(1))));
        });
    }
    // Tensor element-wise mish
    mish() {
        return this.elementWiseSelfDAG((a) => a * Math.tanh(Math.log1p(Math.exp(a))), (self, outGrad) => {
            const tanhSoftPlus = self.exp().add(1).log().tanh();
            // tanh(softplus(x)) + x * (1 - tanh²(softplus(x))) * sigmoid(x)
            const derivative = tanhSoftPlus.add(self.mul(tanhSoftPlus.square().neg().add(1)).mul(self.sigmoid()));
            return outGrad.mul(derivative);
        });
    }
    // Tensor element-wise maximum
    maximum(other) {
        return this.elementWiseABDAG(other, (a, b) => Math.max(a, b), (self, other, outGrad) => outGrad.mul(self.gt(other).add(self.eq(other).mul(0.5))), (self, other, outGrad) => outGrad.mul(other.gt(self).add(other.eq(self).mul(0.5))));
    }
    // Tensor element-wise minimum
    minimum(other) {
        return this.elementWiseABDAG(other, (a, b) => Math.min(a, b), (self, other, outGrad) => outGrad.mul(self.lt(other).add(self.eq(other).mul(0.5))), (self, other, outGrad) => outGrad.mul(other.lt(self).add(other.eq(self).mul(0.5))));
    }
    // Tensor element-wise round
    round() {
        return this.elementWiseSelfDAG((a) => Math.round(a));
    }
    // Tensor element-wise floor
    floor() {
        return this.elementWiseSelfDAG((a) => Math.floor(a));
    }
    // Tensor element-wise ceil
    ceil() {
        return this.elementWiseSelfDAG((a) => Math.ceil(a));
    }
    // Tensor element-wise truncation
    trunc() {
        return this.elementWiseSelfDAG((a) => Math.trunc(a));
    }
    fix = this.trunc;
    // Tensor element-wise fraction portion
    frac() {
        return this.elementWiseSelfDAG((a) => a - Math.floor(a));
    }
    // Tensor element-wise clip and clamp
    clip(min, max) {
        return this.elementWiseSelfDAG((a) => Math.max(min, Math.min(max, a)), (self, outGrad) => outGrad.mul(self.ge(min).mul(self.le(max))));
    }
    clamp = this.clip;
    // Tensor element-wise error function
    erf() {
        return this.elementWiseSelfDAG((a) => (0, utils_1.erf)(a), (self, outGrad) => outGrad.mul(self.square().neg().exp().mul(2 / Math.sqrt(Math.PI))));
    }
    // Tensor element-wise complementary error function
    erfc() {
        return this.elementWiseSelfDAG((a) => (0, utils_1.erfc)(a), (self, outGrad) => outGrad.mul(self.square().neg().exp().mul(2 / Math.sqrt(Math.PI)).neg()));
    }
    // Tensor element-wise inverse error function
    erfinv() {
        return this.elementWiseSelfDAG((a) => (0, utils_1.erfinv)(a), (self, outGrad) => outGrad.mul(self.erfinv().square().exp().mul(Math.sqrt(Math.PI) / 2)));
    }
    // Transpose
    transpose(dim1, dim2) {
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
        const out = new Tensor(this.value, { shape: newShape, strides: newStrides });
        out.requiresGrad = this.requiresGrad;
        // Handle gradient if needed
        if (this.requiresGrad) {
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, out.grad.withGrad(false).transpose(dim1, dim2));
            };
        }
        return out;
    }
    swapaxes = this.transpose;
    swapdims = this.transpose;
    // Transpose 2D
    t() {
        // Verify matrix shape
        if (this.shape.length !== 2) {
            throw new Error("Input is not a matrix");
        }
        return this.transpose(0, 1);
    }
    // 1D tensor dot product
    dot(other) {
        other = Tensor.forceTensor(other);
        // Verify 1D shape
        if (this.shape.length !== 1 || other.shape.length !== 1) {
            throw new Error("Inputs are not 1D tensors");
        }
        // Simple vector dot product
        const vectLen = this.shape[0];
        const vectA = this.value;
        const vectB = other.value;
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
                const outGrad = out.grad.withGrad(false);
                const selfNoGrad = this.withGrad(false);
                const otherNoGrad = other.withGrad(false);
                if (this.requiresGrad)
                    Tensor.addGrad(this, outGrad.mul(otherNoGrad));
                if (other.requiresGrad)
                    Tensor.addGrad(other, outGrad.mul(selfNoGrad));
            };
        }
        return out;
    }
    // Matrix multiplication
    mm(other) {
        other = Tensor.forceTensor(other);
        // Verify 2D shape
        if (this.shape.length !== 2 || other.shape.length !== 2) {
            throw new Error("Inputs are not matrices");
        }
        // Simple matrix multiplication
        const matA = this.value;
        const matB = other.value;
        const matAStrides = this.strides;
        const matBStrides = other.strides;
        const matARows = this.shape[0];
        const matACols = this.shape[1];
        const matBRows = other.shape[0];
        const matBCols = other.shape[1];
        if (matACols !== matBRows)
            throw new Error("Invalid matrices shape for multiplication");
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
                const outGrad = out.grad.withGrad(false);
                const selfNoGrad = this.withGrad(false);
                const otherNoGrad = other.withGrad(false);
                if (this.requiresGrad)
                    Tensor.addGrad(this, outGrad.mm(otherNoGrad.t()));
                if (other.requiresGrad)
                    Tensor.addGrad(other, selfNoGrad.t().mm(outGrad));
            };
        }
        return out;
    }
    // Batched 3D tensor matmul
    bmm(other) {
        other = Tensor.forceTensor(other);
        // Verify 3D shape
        if (this.shape.length !== 3 || other.shape.length !== 3 || this.shape[0] !== other.shape[0]) {
            throw new Error("Inputs are not 3D tensors with the same first dim size");
        }
        // Simple matrix multiplication
        const batchA = this.value;
        const batchB = other.value;
        const batchAStrides = this.strides;
        const batchBStrides = other.strides;
        const batchSize = this.shape[0];
        const batchARows = this.shape[1];
        const batchACols = this.shape[2];
        const batchBRows = other.shape[1];
        const batchBCols = other.shape[2];
        if (batchACols !== batchBRows)
            throw new Error("Invalid matrices shape for multiplication");
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
                const outGrad = out.grad.withGrad(false);
                const selfNoGrad = this.withGrad(false);
                const otherNoGrad = other.withGrad(false);
                if (this.requiresGrad)
                    Tensor.addGrad(this, outGrad.bmm(otherNoGrad.transpose(1, 2)));
                if (other.requiresGrad)
                    Tensor.addGrad(other, selfNoGrad.transpose(1, 2).bmm(outGrad));
            };
        }
        return out;
    }
    // Convert right-side 1D tensor to a vector (nx1 tensor) to do matmul
    mv(other) {
        other = Tensor.forceTensor(other);
        // Verify 2D shape
        if (this.shape.length !== 2 || other.shape.length !== 1) {
            throw new Error("Input is not a 2D and 1D tensor pair");
        }
        // MM with no grad
        const thisMat = new Tensor(this.value, { shape: this.shape, strides: this.strides });
        const otherMat = new Tensor(other.value, { shape: [other.shape[0], 1], strides: [other.strides[0], 1] });
        const out = thisMat.mm(otherMat).squeeze(1);
        // Handle grad with original tensors
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
                const outGrad = out.grad.withGrad(false);
                const selfNoGrad = this.withGrad(false);
                const otherNoGrad = other.withGrad(false);
                if (this.requiresGrad)
                    Tensor.addGrad(this, outGrad.unsqueeze(1).mm(otherNoGrad.unsqueeze(0)));
                if (other.requiresGrad)
                    Tensor.addGrad(other, selfNoGrad.t().mv(outGrad));
            };
        }
        return out;
    }
    // General matrix multiplication with different shapes
    matmul(other) {
        other = Tensor.forceTensor(other);
        if (this.shape.length === 1 && other.shape.length === 1) {
            return this.dot(other);
        }
        else if (this.shape.length === 1 && other.shape.length === 2) {
            return this.unsqueeze(0).mm(other).squeeze(0);
        }
        else if (this.shape.length === 2 && other.shape.length === 1) {
            return this.mv(other);
        }
        else if (this.shape.length === 2 && other.shape.length === 2) {
            return this.mm(other);
        }
        // Too lazy for batched matmul
        throw new Error(`Shapes [${this.shape}] and [${other.shape}] are not supported`);
    }
    // Utility to create a new tensor filled with a number
    static full(shape, num, options = {}) {
        if (shape.length === 0)
            return new Tensor(num, options);
        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize).fill(num);
        return new Tensor(outputValue, { shape, ...options });
    }
    // Utility to create a new tensor with shape of another tensor, filled with a number
    static fullLike(tensor, num, options = {}) {
        if (typeof tensor.value === "number")
            return new Tensor(num, options);
        return new Tensor(new Array(tensor.value.length).fill(num), { shape: tensor.shape, strides: tensor.strides, ...options });
    }
    // Utility to create a new tensor filled with 1
    static ones(shape, options = {}) {
        if (typeof shape === "undefined" || shape.length === 0)
            return new Tensor(1, options);
        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize).fill(1);
        return new Tensor(outputValue, { shape, ...options });
    }
    // Utility to create a new tensor with shape of another tensor, filled with 1
    static onesLike(tensor, options = {}) {
        if (typeof tensor.value === "number")
            return new Tensor(1, options);
        return new Tensor(new Array(tensor.value.length).fill(1), { shape: tensor.shape, strides: tensor.strides, ...options });
    }
    // Utility to create a new tensor filled with 0
    static zeros(shape, options = {}) {
        if (typeof shape === "undefined" || shape.length === 0)
            return new Tensor(0, options);
        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize).fill(0);
        return new Tensor(outputValue, { shape, ...options });
    }
    // Utility to create a new tensor with shape of another tensor, filled with 0
    static zerosLike(tensor, options = {}) {
        if (typeof tensor.value === "number")
            return new Tensor(0, options);
        return new Tensor(new Array(tensor.value.length).fill(0), { shape: tensor.shape, strides: tensor.strides, ...options });
    }
    // Utility to create a new tensor filled with a random number with uniform distribution from 0 to 1
    static rand(shape, options = {}) {
        if (typeof shape === "undefined" || shape.length === 0)
            return new Tensor((0, utils_1.randUniform)(), options);
        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);
        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = (0, utils_1.randUniform)();
        }
        return new Tensor(outputValue, { shape, ...options });
    }
    // Utility to create a new tensor with shape of another tensor, filled with a random number with uniform distribution from 0 to 1
    static randLike(tensor, options = {}) {
        if (typeof tensor.value === "number")
            return new Tensor((0, utils_1.randUniform)(), options);
        const outputValue = new Array(tensor.value.length);
        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = (0, utils_1.randUniform)();
        }
        return new Tensor(outputValue, {
            shape: tensor.shape, strides: tensor.strides, ...options
        });
    }
    // Utility to create a new tensor filled with a random number with normal distribution of mean=0 and stddev=1
    static randn(shape, options = {}) {
        if (typeof shape === "undefined" || shape.length === 0)
            return new Tensor((0, utils_1.randNormal)(), options);
        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);
        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = (0, utils_1.randNormal)();
        }
        return new Tensor(outputValue, { shape, ...options });
    }
    // Utility to create a new tensor with shape of another tensor, filled with a random number with normal distribution of mean=0 and stddev=1
    static randnLike(tensor, options = {}) {
        if (typeof tensor.value === "number")
            return new Tensor((0, utils_1.randNormal)(), options);
        const outputValue = new Array(tensor.value.length);
        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = (0, utils_1.randNormal)();
        }
        return new Tensor(outputValue, {
            shape: tensor.shape, strides: tensor.strides, ...options
        });
    }
    // Utility to create a new tensor filled with a random integer between low and high
    static randint(shape, low, high, options = {}) {
        if (shape.length === 0)
            return new Tensor((0, utils_1.randInt)(low, high), options);
        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);
        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = (0, utils_1.randInt)(low, high);
        }
        return new Tensor(outputValue, { shape, ...options });
    }
    // Utility to create a new tensor with shape of another tensor, filled with a random integer between low and high
    static randintLike(tensor, low, high, options = {}) {
        if (typeof tensor.value === "number")
            return new Tensor((0, utils_1.randInt)(low, high), options);
        const outputValue = new Array(tensor.value.length);
        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = (0, utils_1.randInt)(low, high);
        }
        return new Tensor(outputValue, {
            shape: tensor.shape, strides: tensor.strides, ...options
        });
    }
    // Utility to create a new tensor filled with a random number with normal distribution of custom mean and stddev
    static normal(shape, mean, stdDev, options = {}) {
        if (shape.length === 0)
            return new Tensor((0, utils_1.randNormal)(mean, stdDev), options);
        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);
        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = (0, utils_1.randNormal)(mean, stdDev);
        }
        return new Tensor(outputValue, { shape, ...options });
    }
    // Utility to create a new tensor filled with a random number with uniform distribution from low to high
    static uniform(shape, low, high, options = {}) {
        if (shape.length === 0)
            return new Tensor((0, utils_1.randUniform)(low, high), options);
        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize);
        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = (0, utils_1.randUniform)(low, high);
        }
        return new Tensor(outputValue, { shape, ...options });
    }
    // Reverse-mode autodiff call
    backward() {
        // Build topological order
        const topo = [];
        const visited = new Set();
        function build(node) {
            if (!visited.has(node) && node.requiresGrad) {
                visited.add(node);
                node.grad = Tensor.zerosLike(node); // Reset grad with 0
                for (let child of node.children)
                    build(child);
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
    val() {
        if (typeof this.value === "number")
            return this.value;
        function buildNested(data, shape, strides, baseIndex = 0, dim = 0) {
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
    // Returns a copy of the tensor with gradient turned on/off and detaches from autograd
    withGrad(requiresGrad) {
        return new Tensor(this.value, {
            shape: this.shape,
            strides: this.strides,
            requiresGrad
        });
    }
}
exports.Tensor = Tensor;

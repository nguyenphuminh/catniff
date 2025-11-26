"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Tensor = void 0;
const dtype_1 = require("./dtype");
const utils_1 = require("./utils");
class Tensor {
    value;
    shape;
    strides;
    offset;
    numel;
    grad;
    requiresGrad;
    gradFn;
    children;
    device;
    dtype;
    static training = false;
    static noGrad = false;
    static createGraph = false;
    constructor(value, options = {}) {
        // Memory buffer
        this.dtype = options.dtype || "float32";
        const flatValue = Tensor.flattenValue(value);
        const TypedArrayConstructor = dtype_1.TypedArray[this.dtype];
        this.value = flatValue instanceof TypedArrayConstructor ? flatValue : TypedArrayConstructor.from(flatValue);
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
    static flattenValue(tensorValue) {
        // Handle scalar tensors
        if (typeof tensorValue === "number")
            return [tensorValue];
        // If value is already 1D, we just need to return the value ('s reference)
        if (typeof tensorValue[0] === "number")
            return tensorValue;
        // Or else recursively traverse through the nD array to flatten
        const result = [];
        function traverse(arr) {
            if (typeof arr === "number") {
                result.push(arr);
                // Assume if we can index a value, it is an ArrayLike
            }
            else if (typeof arr[0] !== "undefined") {
                for (let index = 0; index < arr.length; index++) {
                    traverse(arr[index]);
                }
            }
        }
        traverse(tensorValue);
        return result;
    }
    // Utility to get shape from tensor *value*
    static getShape(tensorValue) {
        const shape = [];
        let subA = tensorValue;
        while (typeof subA !== "number") {
            shape.push(subA.length);
            subA = subA[0];
        }
        return shape;
    }
    // Utility to get strides from shape
    static getStrides(shape) {
        if (shape.length === 0)
            return [];
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
                throw new Error(`Can not broadcast shapes: ${shapeA} and ${shapeB}`);
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
    // Utility to get best possible result type if type conflicts happen:
    static getResultDtype(type1, type2) {
        if (type1 === type2)
            return type1;
        const type1Ranking = dtype_1.dtypeHiearchy[type1];
        const type2Ranking = dtype_1.dtypeHiearchy[type2];
        if (type1Ranking > type2Ranking) {
            return type1;
        }
        return type2;
    }
    // Utility to handle other tensor if an op needs a second operand
    handleOther(other) {
        if (other instanceof Tensor) {
            if (this.device !== other.device) {
                throw new Error("Can not operate on tensors that are not on the same device");
            }
            return other;
        }
        return new Tensor(other, {
            offset: 0,
            device: this.device,
            dtype: this.dtype
        });
    }
    // Utility for binary (two operators involved) element-wise ops
    static elementWiseAB(tA, tB, op) {
        const outputDtype = Tensor.getResultDtype(tA.dtype, tB.dtype);
        // Both are scalars
        if (tA.shape.length === 0 && tB.shape.length === 0) {
            return new Tensor(op(tA.value[0], tB.value[0]), {
                shape: [],
                strides: [],
                offset: 0,
                numel: 1,
                device: tA.device,
                dtype: outputDtype
            });
        }
        // First tensor is scalar
        if (tA.shape.length === 0) {
            return Tensor.elementWiseSelf(tB, (a) => op(a, tA.value[0]));
        }
        // Second tensor is scalar
        if (tB.shape.length === 0) {
            return Tensor.elementWiseSelf(tA, (a) => op(a, tB.value[0]));
        }
        // Pad + broadcast shape
        const [paddedAStrides, paddedBStrides, paddedAShape, paddedBShape] = Tensor.padShape(tA.strides, tB.strides, tA.shape, tB.shape);
        const outputShape = Tensor.broadcastShapes(paddedAShape, paddedBShape);
        // Get other output info
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new dtype_1.TypedArray[outputDtype](outputSize);
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
            offset: 0,
            numel: outputSize,
            device: tA.device,
            dtype: outputDtype
        });
    }
    // Utility for self-inflicting element-wise ops
    static elementWiseSelf(tA, op) {
        // Handle scalar case
        if (tA.shape.length === 0)
            return new Tensor(op(tA.value[0]), {
                shape: [],
                strides: [],
                offset: 0,
                numel: 1,
                device: tA.device,
                dtype: tA.dtype
            });
        const contiguous = tA.isContiguous();
        const outputShape = tA.shape;
        const outputStrides = contiguous ? tA.strides : Tensor.getStrides(outputShape);
        const outputSize = tA.numel;
        const outputValue = new dtype_1.TypedArray[tA.dtype](outputSize);
        if (contiguous) {
            for (let index = 0; index < outputSize; index++) {
                outputValue[index] = op(tA.value[index + tA.offset]);
            }
        }
        else {
            for (let index = 0; index < outputSize; index++) {
                const outputCoords = Tensor.indexToCoords(index, outputStrides);
                const originalIndex = tA.offset + Tensor.coordsToIndex(outputCoords, tA.strides);
                outputValue[index] = op(tA.value[originalIndex]);
            }
        }
        return new Tensor(outputValue, {
            shape: outputShape,
            strides: outputStrides,
            offset: 0,
            numel: tA.numel,
            device: tA.device,
            dtype: tA.dtype
        });
    }
    // Utility to do element-wise operation and build a dag node with another tensor
    elementWiseABDAG(other, op, thisGrad = () => new Tensor(0), otherGrad = () => new Tensor(0)) {
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
                const outGrad = out.grad;
                const selfWithGrad = Tensor.createGraph ? this : this.detach();
                const otherWithGrad = Tensor.createGraph ? other : other.detach();
                if (this.requiresGrad)
                    Tensor.addGrad(this, thisGrad(selfWithGrad, otherWithGrad, outGrad));
                if (other.requiresGrad)
                    Tensor.addGrad(other, otherGrad(selfWithGrad, otherWithGrad, outGrad));
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
                const outGrad = out.grad;
                const selfWithGrad = Tensor.createGraph ? this : this.detach();
                if (this.requiresGrad)
                    Tensor.addGrad(this, thisGrad(selfWithGrad, outGrad));
            };
        }
        return out;
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
            tensor.grad = tensor.grad.add(squeezedGrad.cast(tensor.dtype));
        }
    }
    static normalizeDims(dims, numDims) {
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
    isContiguous() {
        const expectedStrides = Tensor.getStrides(this.shape);
        for (let i = 0; i < this.strides.length; i++) {
            if (this.strides[i] !== expectedStrides[i]) {
                return false;
            }
        }
        return true;
    }
    contiguous() {
        // Check if scalar
        if (this.shape.length === 0)
            return this;
        // Check if already contiguous
        if (this.isContiguous())
            return this;
        const outputStrides = Tensor.getStrides(this.shape);
        const outputSize = this.numel;
        const outputValue = new dtype_1.TypedArray[this.dtype](outputSize);
        for (let index = 0; index < outputSize; index++) {
            const outputCoords = Tensor.indexToCoords(index, outputStrides);
            const originalIndex = Tensor.coordsToIndex(outputCoords, this.strides);
            outputValue[index] = this.value[this.offset + originalIndex];
        }
        const out = new Tensor(outputValue, {
            shape: this.shape,
            strides: outputStrides,
            offset: 0,
            numel: outputSize,
            device: this.device,
            dtype: this.dtype
        });
        // Gradient flow back to the original tensor
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, out.grad);
            };
        }
        return out;
    }
    view(newShape) {
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
        const out = new Tensor(this.value, {
            shape: newShape,
            strides: outputStrides,
            offset: this.offset,
            numel: outputSize,
            device: this.device,
            dtype: this.dtype
        });
        // Gradient reshaped and flow back to the original tensor
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, out.grad.reshape(this.shape));
            };
        }
        return out;
    }
    reshape(newShape) {
        return this.contiguous().view(newShape);
    }
    flatten(startDim = 0, endDim = -1) {
        // Handle negative indices
        if (startDim < 0) {
            startDim += this.shape.length;
        }
        if (endDim < 0) {
            endDim += this.shape.length;
        }
        // If dimension out of bound, throw error
        if (startDim >= this.shape.length || endDim >= this.shape.length || startDim < 0 || endDim < 0) {
            throw new Error("Dimensions do not exist to flatten");
        }
        const newShape = [];
        let middleSize = 1;
        for (let index = 0; index < this.shape.length; index++) {
            // Keep dims before startDim
            if (index < startDim) {
                newShape.push(this.shape[index]);
            }
            // Multiply dims from startDim to endDim
            if (index >= startDim && index <= endDim) {
                middleSize *= this.shape[index];
            }
            // Push new flatten middle
            if (index === endDim) {
                newShape.push(middleSize);
            }
            // Keep dims after endDim
            if (index > endDim) {
                newShape.push(this.shape[index]);
            }
        }
        return this.reshape(newShape);
    }
    // Transpose
    transpose(dim1, dim2) {
        // Handle negative indices
        if (dim1 < 0) {
            dim1 += this.shape.length;
        }
        if (dim2 < 0) {
            dim2 += this.shape.length;
        }
        // If dimension out of bound, throw error
        if (dim1 >= this.shape.length || dim2 >= this.shape.length || dim1 < 0 || dim2 < 0) {
            throw new Error("Dimensions do not exist to transpose");
        }
        // If same dimension, return view
        if (dim1 === dim2)
            return this;
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
            device: this.device,
            dtype: this.dtype
        });
        out.requiresGrad = this.requiresGrad;
        // Handle gradient if needed
        if (this.requiresGrad) {
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, out.grad.transpose(dim1, dim2));
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
    // Permute
    permute(dims) {
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
            device: this.device,
            dtype: this.dtype
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
                const permutedGrad = out.grad.permute(inverseAxes);
                Tensor.addGrad(this, permutedGrad);
            };
        }
        return out;
    }
    // Utility for indexing with array of indices
    indexWithArray(indices) {
        if (this.shape.length === 0)
            return this;
        indices = Tensor.normalizeDims(indices, this.shape[0]);
        // Init necessary stuff for indexing
        const reducedShape = this.shape.slice(1);
        const reducedStrides = this.strides.slice(1);
        const elementsPerIndex = Tensor.shapeToSize(reducedShape);
        // Init output data
        const outputShape = [indices.length, ...reducedShape];
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new dtype_1.TypedArray[this.dtype](outputSize);
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
            offset: 0,
            numel: outputSize,
            device: this.device,
            dtype: this.dtype
        });
        // Handle gradient
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                const outGrad = out.grad;
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
                        grad.value[targetIndex] += outGrad.value[sourceStart + j];
                    }
                }
                Tensor.addGrad(this, grad);
            };
        }
        return out;
    }
    // Tensor indexing
    index(indices) {
        const tensorIndices = this.handleOther(indices).clone();
        if (tensorIndices.shape.length === 0) {
            return this.indexWithArray([tensorIndices.value[0]]).squeeze(0);
        }
        else {
            const originalShape = tensorIndices.shape;
            const flatIndices = tensorIndices.value;
            const result = this.indexWithArray(Array.from(flatIndices));
            // Reshape to preserve input shape
            const outputShape = [...originalShape, ...this.shape.slice(1)];
            return result.reshape(outputShape);
        }
    }
    // Tensor slicing
    slice(ranges) {
        // Handle scalars
        if (this.shape.length === 0)
            return this;
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
            if (start < 0)
                start += dimSize;
            if (end < 0)
                end += dimSize;
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
            device: this.device,
            dtype: this.dtype
        });
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                // Create zero tensor of original shape
                const grad = Tensor.zerosLike(this);
                // Upstream grad
                const outGrad = out.grad;
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
                    grad.value[targetIndex] += outGrad.value[srcIndex];
                }
                Tensor.addGrad(this, grad);
            };
        }
        return out;
    }
    // Tensor chunk
    chunk(chunks, dim = 0) {
        // Handle negative indices
        if (dim < 0) {
            dim += this.shape.length;
        }
        // If dimension out of bound, throw error
        if (dim >= this.shape.length || dim < 0) {
            throw new Error("Dimension do not exist to chunk");
        }
        const sliceOpt = new Array(this.shape.length);
        for (let index = 0; index < sliceOpt.length; index++) {
            sliceOpt[index] = [];
        }
        const dimSize = this.shape[dim];
        const chunkDimSize = Math.ceil(dimSize / chunks);
        const results = [];
        for (let index = 0; index < dimSize; index += chunkDimSize) {
            sliceOpt[dim] = [index, Math.min(index + chunkDimSize, dimSize)];
            results.push(this.slice(sliceOpt));
        }
        return results;
    }
    // Tensor expansion
    expand(newShape) {
        // Handle scalars
        let self = this;
        if (this.shape.length === 0) {
            self = self.unsqueeze(0);
        }
        // Pad shapes to same length
        const ndim = Math.max(self.shape.length, newShape.length);
        const oldShape = [...Array(ndim - self.shape.length).fill(1), ...self.shape];
        const oldStrides = [...Array(ndim - self.strides.length).fill(0), ...self.strides];
        const targetShape = [...Array(ndim - newShape.length).fill(1), ...newShape];
        const newStrides = new Array(ndim);
        for (let i = 0; i < ndim; i++) {
            if (oldShape[i] === targetShape[i]) {
                newStrides[i] = oldStrides[i];
            }
            else if (oldShape[i] === 1) {
                newStrides[i] = 0;
            }
            else {
                throw new Error(`Cannot expand dimension of size ${oldShape[i]} to ${targetShape[i]}`);
            }
        }
        const out = new Tensor(self.value, {
            shape: targetShape,
            strides: newStrides,
            offset: self.offset,
            device: self.device,
            dtype: self.dtype
        });
        if (self.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(self);
            out.gradFn = () => {
                Tensor.addGrad(self, out.grad);
            };
        }
        return out;
    }
    // Tensor concatentation
    cat(other, dim = 0) {
        other = this.handleOther(other);
        // Handle scalars
        if (this.shape.length === 0 || other.shape.length === 0) {
            throw new Error("Can not concatenate scalars");
        }
        // Handle negative indices
        if (dim < 0) {
            dim += this.shape.length;
        }
        // If dimension out of bound, throw error
        if (dim >= this.shape.length || dim < 0) {
            throw new Error("Dimension does not exist to concatenate");
        }
        // If shape does not match, throw error
        if (this.shape.length !== other.shape.length) {
            throw new Error("Shape does not match to concatenate");
        }
        const outputShape = new Array(this.shape.length);
        for (let currentDim = 0; currentDim < this.shape.length; currentDim++) {
            if (currentDim === dim) {
                outputShape[currentDim] = this.shape[currentDim] + other.shape[currentDim];
            }
            else if (this.shape[currentDim] !== other.shape[currentDim]) {
                throw new Error("Shape does not match to concatenate");
            }
            else {
                outputShape[currentDim] = this.shape[currentDim];
            }
        }
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputDtype = Tensor.getResultDtype(this.dtype, other.dtype);
        const outputValue = new dtype_1.TypedArray[outputDtype](outputSize);
        for (let outIndex = 0; outIndex < outputSize; outIndex++) {
            const coords = Tensor.indexToCoords(outIndex, outputStrides);
            // Check which tensor this output position comes from
            if (coords[dim] < this.shape[dim]) {
                // Comes from this tensor
                const srcIndex = Tensor.coordsToIndex(coords, this.strides);
                outputValue[outIndex] = this.value[srcIndex + this.offset];
            }
            else {
                // Comes from other tensor - adjust coordinate in concat dimension
                const otherCoords = [...coords];
                otherCoords[dim] -= this.shape[dim];
                const srcIndex = Tensor.coordsToIndex(otherCoords, other.strides);
                outputValue[outIndex] = other.value[srcIndex + other.offset];
            }
        }
        const out = new Tensor(outputValue, {
            shape: outputShape,
            strides: outputStrides,
            offset: 0,
            numel: outputSize,
            device: this.device,
            dtype: this.dtype
        });
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
                const outGrad = out.grad;
                const thisRanges = new Array(this.shape.length);
                const otherRanges = new Array(other.shape.length);
                for (let currentDim = 0; currentDim < this.shape.length; currentDim++) {
                    if (currentDim === dim) {
                        thisRanges[currentDim] = [0, this.shape[currentDim], 1];
                        otherRanges[currentDim] = [this.shape[currentDim], outputShape[currentDim], 1];
                    }
                    else {
                        thisRanges[currentDim] = [];
                        otherRanges[currentDim] = [];
                    }
                }
                Tensor.addGrad(this, outGrad.slice(thisRanges));
                Tensor.addGrad(other, outGrad.slice(otherRanges));
            };
        }
        return out;
    }
    // Tensor squeeze
    squeeze(dims) {
        if (this.shape.length === 0)
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
        dims = Tensor.normalizeDims(dims, this.shape.length);
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
        const outValue = outShape.length === 0 ? this.value[this.offset] : this.value;
        const out = new Tensor(outValue, {
            shape: outShape,
            strides: outStrides,
            offset: this.offset,
            numel: this.numel,
            device: this.device,
            dtype: this.dtype
        });
        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                let restoredGrad = out.grad;
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
        // Handle negative indices
        if (dim < 0) {
            dim += this.shape.length;
        }
        let thisValue = this.value;
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
        const out = new Tensor(thisValue, {
            shape: newShape,
            strides: newStrides,
            offset: this.offset,
            numel: this.numel,
            device: this.device,
            dtype: this.dtype
        });
        // Set up gradient if needed
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, out.grad.squeeze(dim));
            };
        }
        return out;
    }
    // Generic reduction operation handler
    static reduce(tensor, dims, keepDims, config) {
        if (tensor.shape.length === 0)
            return tensor;
        if (typeof dims === "undefined") {
            dims = new Array(tensor.shape.length);
            for (let index = 0; index < dims.length; index++) {
                dims[index] = index;
            }
        }
        if (Array.isArray(dims)) {
            dims = Tensor.normalizeDims(dims, tensor.shape.length);
            const sortedDims = dims.sort((a, b) => b - a);
            let reducedThis = tensor;
            for (let i = 0; i < sortedDims.length; i++) {
                reducedThis = Tensor.reduce(reducedThis, sortedDims[i], true, config);
            }
            return keepDims ? reducedThis : reducedThis.squeeze(dims);
        }
        const outputShape = tensor.shape.map((dim, i) => dims === i ? 1 : dim);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = Tensor.shapeToSize(outputShape);
        const outputValue = new dtype_1.TypedArray[tensor.dtype](outputSize).fill(config.identity);
        const outputCounters = config.needsCounters ? new dtype_1.TypedArray[tensor.dtype](outputSize).fill(0) : new dtype_1.TypedArray[tensor.dtype]();
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
        const out = new Tensor(outputValue, {
            shape: outputShape,
            strides: outputStrides,
            offset: 0,
            numel: outputSize,
            device: tensor.device,
            dtype: tensor.dtype
        });
        // Gradient setup
        if (tensor.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(tensor);
            out.gradFn = () => {
                let shareCounts = new dtype_1.TypedArray[tensor.dtype]();
                if (config.needsShareCounts) {
                    shareCounts = new dtype_1.TypedArray[tensor.dtype](outputSize).fill(0);
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
                const gradValue = new dtype_1.TypedArray[tensor.dtype](originalSize);
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
                        originalValue: tensor.value,
                        counters: outputCounters,
                        shareCounts,
                        realIndex: realFlatIndex,
                        outIndex: outFlatIndex
                    });
                }
                const localGrad = new Tensor(gradValue, {
                    shape: tensor.shape,
                    offset: 0,
                    numel: tensor.numel,
                    device: tensor.device,
                    dtype: tensor.dtype
                });
                Tensor.addGrad(tensor, out.grad.mul(localGrad));
            };
        }
        return keepDims ? out : out.squeeze(dims);
    }
    // Simplified reduction operations
    sum(dims, keepDims = false) {
        return Tensor.reduce(this, dims, keepDims, {
            identity: 0,
            operation: (a, b) => a + b,
            gradientFn: ({}) => 1
        });
    }
    prod(dims, keepDims = false) {
        return Tensor.reduce(this, dims, keepDims, {
            identity: 1,
            operation: (a, b) => a * b,
            gradientFn: ({ outputValue, originalValue, realIndex, outIndex }) => outputValue[outIndex] / originalValue[realIndex]
        });
    }
    mean(dims, keepDims = false) {
        return Tensor.reduce(this, dims, keepDims, {
            identity: 0,
            operation: (a, b) => a + b,
            needsCounters: true,
            postProcess: ({ values, counters }) => {
                for (let i = 0; i < values.length; i++) {
                    values[i] /= counters[i];
                }
            },
            gradientFn: ({ counters, outIndex }) => 1 / counters[outIndex]
        });
    }
    max(dims, keepDims = false) {
        return Tensor.reduce(this, dims, keepDims, {
            identity: -Infinity,
            operation: (a, b) => Math.max(a, b),
            needsShareCounts: true,
            gradientFn: ({ outputValue, originalValue, shareCounts, realIndex, outIndex }) => outputValue[outIndex] === originalValue[realIndex] ? 1 / shareCounts[outIndex] : 0
        });
    }
    min(dims, keepDims = false) {
        return Tensor.reduce(this, dims, keepDims, {
            identity: Infinity,
            operation: (a, b) => Math.min(a, b),
            needsShareCounts: true,
            gradientFn: ({ outputValue, originalValue, shareCounts, realIndex, outIndex }) => outputValue[outIndex] === originalValue[realIndex] ? 1 / shareCounts[outIndex] : 0
        });
    }
    // Tensor all condition reduction
    all(dims, keepDims = false) {
        return this.min(dims, keepDims).ne(0);
    }
    // Tensor any condition reduction
    any(dims, keepDims = false) {
        return this.max(dims, keepDims).ne(0);
    }
    // Tensor variance reduction
    var(dims, keepDims = false) {
        const meanXSquared = this.square().mean(dims, keepDims);
        const meanXSquaredExpanded = this.mean(dims, keepDims).square();
        return meanXSquared.sub(meanXSquaredExpanded);
    }
    // Tensor standard deviation reduction
    std(dims, keepDims = false) {
        return this.var(dims, keepDims).sqrt();
    }
    // Tensor softmax
    softmax(dim = -1) {
        if (this.shape.length === 0)
            return this;
        // Handle negative indexing
        if (dim < 0) {
            dim += this.shape.length;
        }
        // If dimension out of bound, throw error
        if (dim >= this.shape.length || dim < 0) {
            throw new Error("Dimension do not exist to apply softmax");
        }
        const maxVals = this.max(dim, true);
        const shifted = this.sub(maxVals);
        const expVals = shifted.exp();
        const sumExp = expVals.sum(dim, true);
        return expVals.div(sumExp);
    }
    // Tensor softmin
    softmin(dim = -1) {
        if (this.shape.length === 0)
            return this;
        // Handle negative indexing
        if (dim < 0) {
            dim += this.shape.length;
        }
        // If dimension out of bound, throw error
        if (dim >= this.shape.length || dim < 0) {
            throw new Error("Dimension do not exist to apply softmin");
        }
        const maxVals = this.max(dim, true);
        const shifted = maxVals.sub(this);
        const expVals = shifted.exp();
        const sumExp = expVals.sum(dim, true);
        return expVals.div(sumExp);
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
    // Tensor element-wise leaky relu
    leakyRelu(negativeSlope = 0.01) {
        return this.elementWiseSelfDAG((a) => Math.max(a, 0) + negativeSlope * Math.min(a, 0), (self, outGrad) => {
            return outGrad.mul(self.gt(0).add(self.le(0).mul(negativeSlope)));
        });
    }
    // Tensor element-wise elu
    elu(alpha = 1) {
        return this.elementWiseSelfDAG((a) => a > 0 ? a : alpha * (Math.expm1(a)), (self, outGrad) => {
            return outGrad.mul(self.gt(0).add(self.le(0).mul(self.exp().mul(alpha))));
        });
    }
    // Tensor element-wise selu
    selu() {
        const alpha = 1.6732632423543772848170429916717;
        const scale = 1.0507009873554804934193349852946;
        return this.elementWiseSelfDAG((a) => scale * (a >= 0 ? a : alpha * Math.expm1(a)), (self, outGrad) => {
            return outGrad.mul(self.gt(0).mul(scale).add(self.le(0).mul(self.exp().mul(alpha * scale))));
        });
    }
    // Tensor element-wise celu
    celu(alpha = 1) {
        return this.elementWiseSelfDAG((a) => a >= 0 ? a : alpha * (Math.expm1(a / alpha)), (self, outGrad) => {
            return outGrad.mul(self.gt(0).add(self.le(0).mul(self.div(alpha).exp())));
        });
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
            // tanh(softplus(x)) + x * (1 - tanh^2(softplus(x))) * sigmoid(x)
            const derivative = tanhSoftPlus.add(self.mul(tanhSoftPlus.square().neg().add(1)).mul(self.sigmoid()));
            return outGrad.mul(derivative);
        });
    }
    // Tensor element-wise gelu
    gelu(approximate = "none") {
        if (approximate === "none") {
            return this.elementWiseSelfDAG((a) => 0.5 * a * (1 + (0, utils_1.erf)(a / Math.sqrt(2))), (self, outGrad) => {
                const sqrt2 = Math.sqrt(2);
                const sqrt2OverPi = Math.sqrt(2 / Math.PI);
                const xOverSqrt2 = self.div(sqrt2);
                const erfVal = xOverSqrt2.erf();
                const phi = xOverSqrt2.square().neg().exp().div(sqrt2OverPi);
                const derivative = erfVal.add(1).mul(0.5).add(self.mul(phi));
                return outGrad.mul(derivative);
            });
        }
        else if (approximate === "tanh") {
            return this.elementWiseSelfDAG((a) => 0.5 * a * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (a + 0.044715 * a * a * a))), (self, outGrad) => {
                const sqrt2OverPi = Math.sqrt(2 / Math.PI);
                const c = 0.044715;
                const tanhArg = self.add(self.pow(3).mul(c)).mul(sqrt2OverPi);
                const tanhVal = tanhArg.tanh();
                const sechSquared = tanhVal.square().neg().add(1);
                const term1 = tanhVal.add(1).mul(0.5);
                const term2 = self.mul(sechSquared).mul(sqrt2OverPi).mul(self.square().mul(c * 3).add(1)).mul(0.5);
                const derivative = term1.add(term2);
                return outGrad.mul(derivative);
            });
        }
        throw new Error("Specified approximation does not exist");
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
    // 1D tensor dot product
    dot(other) {
        other = this.handleOther(other);
        // Verify 1D shape
        if (this.shape.length !== 1 || other.shape.length !== 1) {
            throw new Error("Inputs are not 1D tensors");
        }
        return this.mul(other).sum();
    }
    // Matrix multiplication
    mm(other) {
        other = this.handleOther(other);
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
        const matCDtype = Tensor.getResultDtype(this.dtype, other.dtype);
        const matCShape = [matARows, matBCols];
        const matCStrides = Tensor.getStrides(matCShape);
        const matCSize = Tensor.shapeToSize(matCShape);
        const matC = new dtype_1.TypedArray[matCDtype](matCSize).fill(0);
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
        const out = new Tensor(matC, {
            shape: matCShape,
            strides: matCStrides,
            offset: 0,
            numel: matCSize,
            device: this.device,
            dtype: matCDtype
        });
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
                const outGrad = out.grad;
                const selfWithGrad = Tensor.createGraph ? this : this.detach();
                const otherWithGrad = Tensor.createGraph ? other : other.detach();
                if (this.requiresGrad)
                    Tensor.addGrad(this, outGrad.mm(otherWithGrad.t()));
                if (other.requiresGrad)
                    Tensor.addGrad(other, selfWithGrad.t().mm(outGrad));
            };
        }
        return out;
    }
    // Batched 3D tensor matmul
    bmm(other) {
        other = this.handleOther(other);
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
        const batchCDtype = Tensor.getResultDtype(this.dtype, other.dtype);
        const batchCShape = [batchSize, batchARows, batchBCols];
        const batchCStrides = Tensor.getStrides(batchCShape);
        const batchCSize = Tensor.shapeToSize(batchCShape);
        const batchC = new dtype_1.TypedArray[batchCDtype](batchCSize).fill(0);
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
        const out = new Tensor(batchC, {
            shape: batchCShape,
            strides: batchCStrides,
            offset: 0,
            numel: batchCSize,
            device: this.device,
            dtype: batchCDtype
        });
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
                const outGrad = out.grad;
                const selfWithGrad = Tensor.createGraph ? this : this.detach();
                const otherWithGrad = Tensor.createGraph ? other : other.detach();
                if (this.requiresGrad)
                    Tensor.addGrad(this, outGrad.bmm(otherWithGrad.transpose(1, 2)));
                if (other.requiresGrad)
                    Tensor.addGrad(other, selfWithGrad.transpose(1, 2).bmm(outGrad));
            };
        }
        return out;
    }
    // Convert right-side 1D tensor to a vector (nx1 tensor) to do matmul
    mv(other) {
        other = this.handleOther(other);
        // Verify 2D shape
        if (this.shape.length !== 2 || other.shape.length !== 1) {
            throw new Error("Input is not a 2D and 1D tensor pair");
        }
        return this.mm(other.unsqueeze(1)).squeeze(1);
    }
    // General matrix multiplication with different shapes
    matmul(other) {
        other = this.handleOther(other);
        const isThis1D = this.shape.length === 1;
        const isOther1D = other.shape.length === 1;
        if (isThis1D && isOther1D) {
            return this.dot(other);
        }
        else if (isThis1D && other.shape.length === 2) {
            return this.unsqueeze(0).mm(other).squeeze(0);
        }
        else if (this.shape.length === 2 && isOther1D) {
            return this.mv(other);
        }
        else if (this.shape.length === 2 && other.shape.length === 2) {
            return this.mm(other);
        }
        else if ((this.shape.length > 0 && other.shape.length >= 2) ||
            (this.shape.length >= 2 && other.shape.length > 0)) {
            // Append/prepend dims if needed
            const self = isThis1D ? this.unsqueeze(0) : this;
            other = isOther1D ? other.unsqueeze(1) : other;
            // Padding
            const [selfStrides, otherStrides, selfShape, otherShape] = Tensor.padShape(self.strides, other.strides, self.shape, other.shape);
            const lastDim = selfShape.length - 1;
            // Prepare data for broadcasting
            const batchA = self.value;
            const batchB = other.value;
            const batchARows = selfShape[lastDim - 1];
            const batchACols = selfShape[lastDim];
            const batchBRows = otherShape[lastDim - 1];
            const batchBCols = otherShape[lastDim];
            // Verify if can do matmul
            if (batchACols !== batchBRows)
                throw new Error("Invalid matrices shape for multiplication");
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
            const outputDtype = Tensor.getResultDtype(this.dtype, other.dtype);
            const outputShape = [...offsetShape, batchARows, batchBCols];
            const outputStrides = Tensor.getStrides(outputShape);
            const outputSize = Tensor.shapeToSize(outputShape);
            const outputValue = new dtype_1.TypedArray[outputDtype](outputSize).fill(0);
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
            const out = new Tensor(outputValue, {
                shape: outputShape,
                strides: outputStrides,
                offset: 0,
                numel: outputSize,
                device: this.device,
                dtype: outputDtype
            });
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
                    other = other;
                    const outGrad = out.grad;
                    const selfWithGrad = Tensor.createGraph ? self : self.detach();
                    const otherWithGrad = Tensor.createGraph ? other : other.detach();
                    if (this.requiresGrad)
                        Tensor.addGrad(this, outGrad.matmul(otherWithGrad.transpose(-2, -1)));
                    if (other.requiresGrad)
                        Tensor.addGrad(other, selfWithGrad.transpose(-2, -1).matmul(outGrad));
                };
            }
            return out;
        }
        throw new Error(`Shapes [${this.shape}] and [${other.shape}] are not supported`);
    }
    // Dropout
    dropout(rate) {
        if (!Tensor.training || rate === 0)
            return this;
        const keepRate = 1 - rate;
        const uniform = Tensor.randLike(this);
        const mask = uniform.lt(keepRate);
        return this.mul(mask).div(keepRate);
    }
    // Get the upper triangular part with respect to main diagonal
    triu(diagonal = 0) {
        if (this.shape.length < 2) {
            throw new Error("triu requires at least 2 dimensions");
        }
        const maskShape = this.shape.slice(-2);
        const maskStrides = Tensor.getStrides(maskShape);
        const maskSize = Tensor.shapeToSize(maskShape);
        const maskValue = new dtype_1.TypedArray[this.dtype](maskSize).fill(1);
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
            offset: 0,
            numel: maskSize,
            device: this.device,
            dtype: this.dtype
        });
        return this.mul(mask);
    }
    // Get the lower triangular part with respect to main diagonal
    tril(diagonal = 0) {
        if (this.shape.length < 2) {
            throw new Error("triu requires at least 2 dimensions");
        }
        const maskShape = this.shape.slice(-2);
        const maskStrides = Tensor.getStrides(maskShape);
        const maskSize = Tensor.shapeToSize(maskShape);
        const maskValue = new dtype_1.TypedArray[this.dtype](maskSize).fill(0);
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
            offset: 0,
            numel: maskSize,
            device: this.device,
            dtype: this.dtype
        });
        return this.mul(mask);
    }
    // Fill specific positions of this tensor with a value through a mask
    maskedFill(mask, value) {
        mask = this.handleOther(mask);
        return this.mul(mask.logicalNot()).add(mask.mul(value));
    }
    // Utility to create a new tensor filled with a number
    static full(shape, num, options = {}) {
        if (shape.length === 0)
            return new Tensor(num, options);
        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize).fill(num);
        return new Tensor(outputValue, {
            shape,
            offset: 0,
            numel: outputSize,
            ...options
        });
    }
    // Utility to create a new tensor with shape of another tensor, filled with a number
    static fullLike(tensor, num, options = {}) {
        if (tensor.shape.length === 0)
            return new Tensor(num, {
                offset: 0,
                device: tensor.device,
                dtype: tensor.dtype,
                ...options
            });
        return new Tensor(new Array(tensor.numel).fill(num), {
            shape: tensor.shape,
            offset: 0,
            numel: tensor.numel,
            device: tensor.device,
            dtype: tensor.dtype,
            ...options
        });
    }
    // Utility to create a new tensor filled with 1
    static ones(shape, options = {}) {
        if (typeof shape === "undefined" || shape.length === 0)
            return new Tensor(1, options);
        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize).fill(1);
        return new Tensor(outputValue, {
            shape,
            offset: 0,
            numel: outputSize,
            ...options
        });
    }
    // Utility to create a new tensor with shape of another tensor, filled with 1
    static onesLike(tensor, options = {}) {
        if (tensor.shape.length === 0)
            return new Tensor(1, {
                offset: 0,
                device: tensor.device,
                dtype: tensor.dtype,
                ...options
            });
        return new Tensor(new Array(tensor.numel).fill(1), {
            shape: tensor.shape,
            offset: 0,
            numel: tensor.numel,
            device: tensor.device,
            dtype: tensor.dtype,
            ...options
        });
    }
    // Utility to create a new tensor filled with 0
    static zeros(shape, options = {}) {
        if (typeof shape === "undefined" || shape.length === 0)
            return new Tensor(0, options);
        const outputSize = Tensor.shapeToSize(shape);
        const outputValue = new Array(outputSize).fill(0);
        return new Tensor(outputValue, {
            shape,
            offset: 0,
            numel: outputSize,
            ...options
        });
    }
    // Utility to create a new tensor with shape of another tensor, filled with 0
    static zerosLike(tensor, options = {}) {
        if (tensor.shape.length === 0)
            return new Tensor(0, {
                offset: 0,
                device: tensor.device,
                dtype: tensor.dtype,
                ...options
            });
        return new Tensor(new Array(tensor.numel).fill(0), {
            shape: tensor.shape,
            offset: 0,
            numel: tensor.numel,
            device: tensor.device,
            dtype: tensor.dtype,
            ...options
        });
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
        return new Tensor(outputValue, {
            shape,
            offset: 0,
            numel: outputSize,
            ...options
        });
    }
    // Utility to create a new tensor with shape of another tensor, filled with a random number with uniform distribution from 0 to 1
    static randLike(tensor, options = {}) {
        if (tensor.shape.length === 0)
            return new Tensor((0, utils_1.randUniform)(), {
                offset: 0,
                device: tensor.device,
                dtype: tensor.dtype,
                ...options
            });
        const outputValue = new Array(tensor.numel);
        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = (0, utils_1.randUniform)();
        }
        return new Tensor(outputValue, {
            shape: tensor.shape,
            offset: 0,
            numel: tensor.numel,
            device: tensor.device,
            dtype: tensor.dtype,
            ...options
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
        return new Tensor(outputValue, {
            shape,
            offset: 0,
            numel: outputSize,
            ...options
        });
    }
    // Utility to create a new tensor with shape of another tensor, filled with a random number with normal distribution of mean=0 and stddev=1
    static randnLike(tensor, options = {}) {
        if (tensor.shape.length === 0)
            return new Tensor((0, utils_1.randNormal)(), {
                offset: 0,
                device: tensor.device,
                dtype: tensor.dtype,
                ...options
            });
        const outputValue = new Array(tensor.numel);
        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = (0, utils_1.randNormal)();
        }
        return new Tensor(outputValue, {
            shape: tensor.shape,
            offset: 0,
            numel: tensor.numel,
            device: tensor.device,
            dtype: tensor.dtype,
            ...options
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
        return new Tensor(outputValue, {
            shape,
            offset: 0,
            numel: outputSize,
            ...options
        });
    }
    // Utility to create a new tensor with shape of another tensor, filled with a random integer between low and high
    static randintLike(tensor, low, high, options = {}) {
        if (tensor.shape.length === 0)
            return new Tensor((0, utils_1.randInt)(low, high), {
                offset: 0,
                device: tensor.device,
                dtype: tensor.dtype,
                ...options
            });
        const outputValue = new Array(tensor.numel);
        for (let index = 0; index < outputValue.length; index++) {
            outputValue[index] = (0, utils_1.randInt)(low, high);
        }
        return new Tensor(outputValue, {
            shape: tensor.shape,
            offset: 0,
            numel: tensor.numel,
            device: tensor.device,
            dtype: tensor.dtype,
            ...options
        });
    }
    // Utility to create a new tensor filled with integers from 0 to n, randomly shuffled
    static randperm(n, options = {}) {
        const outputValue = new Array(n);
        for (let i = 0; i < n; i++) {
            outputValue[i] = i;
        }
        (0, utils_1.fyShuffle)(outputValue);
        return new Tensor(outputValue, {
            shape: [n],
            offset: 0,
            numel: n,
            ...options
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
        return new Tensor(outputValue, {
            shape,
            offset: 0,
            numel: outputSize,
            ...options
        });
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
        return new Tensor(outputValue, {
            shape,
            offset: 0,
            numel: outputSize, ...options
        });
    }
    // Utility to create an 1D tensor from a range incrementing with "step"
    static arange(start, stop, step = 1, options = {}) {
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
        return new Tensor(outputValue, {
            shape: outputShape,
            offset: 0,
            numel: outputSize,
            ...options
        });
    }
    // Utility to create an 1D tensor from a range evenly spaced out with a given amount of steps
    static linspace(start, stop, steps, options = {}) {
        if (steps <= 0)
            throw new Error("Steps must be positive");
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
        return new Tensor(outputValue, {
            shape: [steps],
            offset: 0,
            numel: steps,
            ...options
        });
    }
    // Utility to create a 2D tensor with its main diagonal filled with 1s and others with 0s
    static eye(n, m = n, options = {}) {
        const outputSize = n * m;
        const outputShape = [n, m];
        const outputStrides = Tensor.getStrides(outputShape);
        const outputValue = new Array(outputSize).fill(0);
        for (let i = 0; i < Math.min(n, m); i++) {
            outputValue[i * outputStrides[0] + i * outputStrides[1]] = 1;
        }
        return new Tensor(outputValue, {
            shape: outputShape,
            offset: 0,
            strides: outputStrides,
            numel: outputSize,
            ...options
        });
    }
    // Reverse-mode autodiff call
    backward(options = {}) {
        // Init
        const zeroGrad = options.zeroGrad ?? true;
        // Build topological order
        const topo = [];
        const visited = new Set();
        function build(node) {
            // Only collects unvisited node and node that requires gradient
            if (!visited.has(node) && node.requiresGrad && !Tensor.noGrad) {
                visited.add(node);
                // Reset grad to zeros if specified
                if (zeroGrad) {
                    node.grad = Tensor.zerosLike(node);
                }
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
        if (this.shape.length === 0)
            return this.value[0];
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
        return buildNested(this.value, this.shape, this.strides, this.offset);
    }
    // Returns a view of the tensor with gradient turned off and detaches from autograd
    detach() {
        return new Tensor(this.value, {
            shape: this.shape,
            strides: this.strides,
            offset: this.offset,
            numel: this.numel,
            device: this.device,
            dtype: this.dtype,
            requiresGrad: false
        });
    }
    // Returns a copy of the tensor (with new data allocation) and keeps grad connection
    clone() {
        let out;
        if (this.shape.length === 0) {
            out = new Tensor(this.value, {
                shape: [],
                strides: [],
                offset: 0,
                numel: 1,
                device: this.device,
                dtype: this.dtype
            });
        }
        else {
            const contiguous = this.isContiguous();
            const outputStrides = contiguous ? this.strides : Tensor.getStrides(this.shape);
            const outputSize = this.numel;
            const outputValue = new dtype_1.TypedArray[this.dtype](outputSize);
            if (contiguous) {
                for (let index = 0; index < outputSize; index++) {
                    outputValue[index] = this.value[this.offset + index];
                }
            }
            else {
                for (let index = 0; index < outputSize; index++) {
                    const outputCoords = Tensor.indexToCoords(index, outputStrides);
                    const originalIndex = Tensor.coordsToIndex(outputCoords, this.strides);
                    outputValue[index] = this.value[this.offset + originalIndex];
                }
            }
            out = new Tensor(outputValue, {
                shape: this.shape,
                strides: outputStrides,
                offset: 0,
                numel: outputSize,
                device: this.device,
                dtype: this.dtype
            });
        }
        if (this.requiresGrad) {
            out.requiresGrad = true;
            out.children.push(this);
            out.gradFn = () => {
                Tensor.addGrad(this, out.grad);
            };
        }
        return out;
    }
    // Returns this tensor with value replaced with the value of another tensor
    replace(other) {
        other = this.handleOther(other);
        // Verify shape
        if (this.shape.length !== other.shape.length) {
            throw new Error("Shape mismatch when trying to do tensor value replacement");
        }
        for (let index = 0; index < this.shape.length; index++) {
            if (this.shape[index] !== other.shape[index]) {
                throw new Error("Shape mismatch when trying to do tensor value replacement");
            }
        }
        // Reassign values
        this.value = other.value;
        this.strides = other.strides;
        this.offset = other.offset;
        this.device = other.device;
        this.dtype = other.dtype;
        return this;
    }
    // Op to return a new tensor casted to another dtype
    cast(dtype) {
        if (this.dtype === dtype)
            return this;
        return new Tensor(this.value, {
            shape: this.shape,
            strides: this.strides,
            offset: this.offset,
            numel: this.numel,
            device: this.device,
            dtype: dtype
        });
    }
    // Holds all available backends
    static backends = new Map();
    // Op to transfer tensor to another device
    to(device) {
        if (device === "cpu")
            return this;
        const backend = Tensor.backends.get(device);
        if (backend && backend.transfer) {
            return backend.transfer(this);
        }
        throw new Error(`No device found to transfer tensor to or a handler is not implemented for device.`);
    }
    // Op to transfer tensor to another device in-place
    to_(device) {
        if (device === "cpu")
            return this;
        const backend = Tensor.backends.get(this.device);
        if (backend && backend.create) {
            backend.create(this);
            return this;
        }
        throw new Error(`No device found to transfer tensor to or a handler is not implemented for device.`);
    }
}
exports.Tensor = Tensor;

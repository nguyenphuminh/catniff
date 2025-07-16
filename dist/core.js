"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Tensor = void 0;
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
        if (typeof tensor === "number")
            return tensor;
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
    // Convert flat index to array of coordinates
    static indexToCoords(index, shape, strides) {
        const coords = new Array(shape.length);
        let remaining = index;
        // Sort dimensions by stride (largest first) for correct decomposition
        const sortedDims = shape.map((_, i) => i).sort((a, b) => strides[b] - strides[a]);
        for (const dim of sortedDims) {
            coords[dim] = Math.floor(remaining / strides[dim]);
            remaining %= strides[dim];
        }
        return coords;
    }
    // Convert array of coordinates to *unbroadcasted* flat index 
    static coordsToIndex(coords, shape, strides) {
        let index = 0;
        for (let i = 0; i < coords.length; i++) {
            const coord = coords[i];
            // Handle broadcasting
            const actualCoord = shape[i] === 1 ? 0 : coord;
            index += actualCoord * strides[i];
        }
        return index;
    }
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
        const outputSize = outputShape.reduce((a, b) => a * b, 1);
        const outputValue = new Array(outputSize);
        for (let i = 0; i < outputSize; i++) {
            // Get coordinates from 1D index
            const coordsOutput = Tensor.indexToCoords(i, outputShape, outputStrides);
            // Convert the coordinates to 1D index of flattened A with respect to A's shape
            const indexA = Tensor.coordsToIndex(coordsOutput, paddedAShape, paddedAStrides);
            // Convert the coordinates to 1D index of flattened B with respect to B's shape
            const indexB = Tensor.coordsToIndex(coordsOutput, paddedBShape, paddedBStrides);
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
        return new Tensor(tA.value.map(el => op(el)), { shape: [...tA.shape], strides: [...tA.strides] });
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
            return new Tensor(this.value);
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
        const outShape = this.shape.filter((dim, i) => {
            const shouldSqueeze = dims.includes(i);
            if (shouldSqueeze && dim !== 1)
                throw new Error(`Can not squeeze dim with size ${dim}`);
            return !shouldSqueeze;
        });
        // Remove strides of size-1 dims
        const outStrides = this.strides.filter((stride, i) => !dims.includes(i));
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
        if (typeof this.value === "number")
            return new Tensor([this.value]);
        if (dim < 0 || dim > this.shape.length) {
            throw new Error(`Invalid dimension ${dim} for unsqueeze`);
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
        const out = new Tensor(this.value, { shape: newShape, strides: newStrides });
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
            return new Tensor(this.value);
        if (typeof dims === "number") {
            dims = [dims];
        }
        if (typeof dims === "undefined") {
            dims = Array.from({ length: this.shape.length }, (_, index) => index);
        }
        const outputShape = this.shape.map((dim, i) => dims.includes(i) ? 1 : dim);
        const outputStrides = Tensor.getStrides(outputShape);
        const outputSize = outputShape.reduce((a, b) => a * b, 1);
        const outputValue = new Array(outputSize).fill(0);
        const originalSize = this.shape.reduce((a, b) => a * b, 1);
        let gradShape, gradStrides, gradValue = [];
        if (this.requiresGrad) {
            gradShape = [...this.shape];
            gradStrides = [...this.strides];
            gradValue = new Array(originalSize).fill(0);
        }
        for (let index = 0; index < originalSize; index++) {
            const coords = Tensor.indexToCoords(index, this.shape, this.strides);
            // Force 0 on reduced axes to collapse into size-1 dims
            const outCoords = coords.map((val, i) => dims.includes(i) ? 0 : val);
            // Convert output coordinates to flat index
            const outFlatIndex = outCoords.reduce((acc, val, i) => acc + val * outputStrides[i], 0);
            // Accumulate
            const realFlatIndex = coords.reduce((acc, val, i) => acc + val * this.strides[i], 0);
            outputValue[outFlatIndex] += this.value[realFlatIndex];
            // Mark for gradient
            if (this.requiresGrad) {
                (gradValue)[realFlatIndex] = 1;
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
    // Tensor element-wise addition
    add(other) {
        return this.elementWiseABDAG(other, (a, b) => a + b, (self, other, outGrad) => outGrad, (self, other, outGrad) => outGrad);
    }
    // Tensor element-wise subtraction
    sub(other) {
        return this.elementWiseABDAG(other, (a, b) => a - b, (self, other, outGrad) => outGrad, (self, other, outGrad) => outGrad.neg());
    }
    // Tensor element-wise multiplication
    mul(other) {
        return this.elementWiseABDAG(other, (a, b) => a * b, (self, other, outGrad) => outGrad.mul(other), (self, other, outGrad) => outGrad.mul(self));
    }
    // Tensor element-wise power
    pow(other) {
        return this.elementWiseABDAG(other, (a, b) => a ** b, (self, other, outGrad) => outGrad.mul(other.mul(self.pow(other.sub(1)))), (self, other, outGrad) => outGrad.mul(self.pow(other).mul(self.log())));
    }
    // Tensor element-wise division
    div(other) {
        return this.elementWiseABDAG(other, (a, b) => a / b, (self, other, outGrad) => outGrad.div(other), (self, other, outGrad) => outGrad.mul(self.neg().div(other.pow(2))));
    }
    // Tensor element-wise greater or equal comparison
    ge(other) {
        return this.elementWiseABDAG(other, (a, b) => a >= b ? 1 : 0);
    }
    // Tensor element-wise less or equal comparison
    le(other) {
        return this.elementWiseABDAG(other, (a, b) => a <= b ? 1 : 0);
    }
    // Tensor element-wise greater-than comparison
    gt(other) {
        return this.elementWiseABDAG(other, (a, b) => a > b ? 1 : 0);
    }
    // Tensor element-wise less-than comparison
    lt(other) {
        return this.elementWiseABDAG(other, (a, b) => a < b ? 1 : 0);
    }
    // Tensor element-wise equality comparison
    eq(other) {
        return this.elementWiseABDAG(other, (a, b) => a === b ? 1 : 0);
    }
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
    // Tensor element-wise absolute
    abs() {
        return this.elementWiseSelfDAG((a) => Math.abs(a), (self, outGrad) => outGrad.mul(self.sign()));
    }
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
        return this.elementWiseSelfDAG((a) => Math.tan(a), (self, outGrad) => outGrad.mul(self.tan().pow(2).add(1)));
    }
    // Tensor element-wise asin
    asin() {
        return this.elementWiseSelfDAG((a) => Math.asin(a), (self, outGrad) => outGrad.div(self.pow(2).neg().add(1).sqrt()));
    }
    // Tensor element-wise acos
    acos() {
        return this.elementWiseSelfDAG((a) => Math.acos(a), (self, outGrad) => outGrad.div(self.pow(2).neg().add(1).sqrt()).neg());
    }
    // Tensor element-wise atan
    atan() {
        return this.elementWiseSelfDAG((a) => Math.atan(a), (self, outGrad) => outGrad.div(self.pow(2).add(1)));
    }
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
        return this.elementWiseSelfDAG((a) => Math.asinh(a), (self, outGrad) => outGrad.div(self.pow(2).add(1).sqrt()));
    }
    // Tensor element-wise acosh
    acosh() {
        return this.elementWiseSelfDAG((a) => Math.acosh(a), (self, outGrad) => outGrad.div(self.add(1).sqrt().mul(self.sub(1).sqrt())));
    }
    // Tensor element-wise atanh
    atanh() {
        return this.elementWiseSelfDAG((a) => Math.atanh(a), (self, outGrad) => outGrad.div(self.pow(2).neg().add(1)));
    }
    // Tensor element-wise square root
    sqrt() {
        return this.elementWiseSelfDAG((a) => Math.sqrt(a), (self, outGrad) => outGrad.div(self.sqrt().mul(2)));
    }
    // Tensor element-wise e^x
    exp() {
        return this.elementWiseSelfDAG((a) => Math.exp(a), (self, outGrad) => outGrad.mul(self.exp()));
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
        return this.elementWiseSelfDAG((a) => Math.max(a, 0), (self, outGrad) => outGrad.mul(self.ge(0)));
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
        return this.elementWiseSelfDAG((a) => Math.tanh(a), (self, outGrad) => outGrad.mul(self.tanh().pow(2).neg().add(1)));
    }
    // Transpose
    transpose(dim1, dim2) {
        // If dimension out of bound, throw error
        if (dim1 >= this.shape.length || dim2 >= this.shape.length || dim1 < 0 || dim2 < 0) {
            throw new Error("Dimensions do not exist to tranpose");
        }
        // If same dimension, return copy
        if (dim1 === dim2) {
            return new Tensor(this.value, { shape: [...this.shape], strides: [...this.strides] });
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
        const matCSize = matCShape.reduce((a, b) => a * b, 1);
        const matC = new Array(matCSize).fill(0);
        for (let i = 0; i < matARows; i++) {
            for (let j = 0; j < matBCols; j++) {
                for (let k = 0; k < matACols; k++) {
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
    // Convert right-side 1D tensor to a vector (nx1 tensor) to do matmul
    mv(other) {
        other = Tensor.forceTensor(other);
        // Verify 2D shape
        if (this.shape.length !== 2 || other.shape.length !== 1) {
            throw new Error("Input is not a 2D and 1D tensor pair");
        }
        // MM with no grad
        const thisMat = new Tensor(this.value, { shape: [...this.shape], strides: [...this.strides] });
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
    // Utility to create a new tensor with shape of another tensor, filled with a number
    static fullLike(tensor, num, options = {}) {
        if (typeof tensor.value === "number")
            return new Tensor(num, options);
        return new Tensor(tensor.value.map(el => num), { shape: [...tensor.shape], strides: [...tensor.strides], ...options });
    }
    // Reverse-mode autodiff call
    backward() {
        // Build topological order
        const topo = [];
        const visited = new Set();
        function build(node) {
            if (!visited.has(node) && node.requiresGrad) {
                visited.add(node);
                node.grad = Tensor.fullLike(node, 0);
                for (let child of node.children)
                    build(child);
                topo.push(node);
            }
        }
        build(this);
        // Feed backward to calculate gradient
        this.grad = Tensor.fullLike(this, 1);
        for (let index = topo.length - 1; index > -1; index--) {
            topo[index].gradFn();
        }
    }
    // Returns the number/nD array form of tensor
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
    // Returns a copy of the tensor with gradient turned on/off
    withGrad(requiresGrad) {
        return new Tensor(this.value, {
            shape: [...this.shape],
            strides: [...this.strides],
            requiresGrad
        });
    }
}
exports.Tensor = Tensor;

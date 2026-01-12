"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.nn = exports.MultiheadAttention = exports.Embedding = exports.RMSNorm = exports.LayerNorm = exports.GroupNorm = exports.InstanceNorm = exports.BatchNorm = exports.LSTMCell = exports.GRUCell = exports.RNNCell = exports.Linear = void 0;
exports.scaledDotProductAttention = scaledDotProductAttention;
const core_1 = require("./core");
function linearTransform(input, weight, bias) {
    let output = input.matmul(weight.t());
    if (bias) {
        output = output.add(bias);
    }
    return output;
}
class Linear {
    weight;
    bias;
    constructor(inFeatures, outFeatures, bias = true, device, dtype) {
        const bound = 1 / Math.sqrt(inFeatures);
        this.weight = core_1.Tensor.uniform([outFeatures, inFeatures], -bound, bound, { requiresGrad: true, device, dtype });
        if (bias) {
            this.bias = core_1.Tensor.uniform([outFeatures], -bound, bound, { requiresGrad: true, device, dtype });
        }
    }
    forward(input) {
        input = this.weight.handleOther(input);
        return linearTransform(input, this.weight, this.bias);
    }
}
exports.Linear = Linear;
function rnnTransform(input, hidden, inputWeight, hiddenWeight, inputBias, hiddenBias) {
    let output = input.matmul(inputWeight.t()).add(hidden.matmul(hiddenWeight.t()));
    if (inputBias) {
        output = output.add(inputBias);
    }
    if (hiddenBias) {
        output = output.add(hiddenBias);
    }
    return output;
}
class RNNCell {
    weightIH;
    weightHH;
    biasIH;
    biasHH;
    constructor(inputSize, hiddenSize, bias = true, device, dtype) {
        const bound = 1 / Math.sqrt(hiddenSize);
        this.weightIH = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHH = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        if (bias) {
            this.biasIH = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHH = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        }
    }
    forward(input, hidden) {
        input = this.weightIH.handleOther(input);
        hidden = this.weightHH.handleOther(hidden);
        return rnnTransform(input, hidden, this.weightIH, this.weightHH, this.biasIH, this.biasHH).tanh();
    }
}
exports.RNNCell = RNNCell;
class GRUCell {
    weightIR;
    weightIZ;
    weightIN;
    weightHR;
    weightHZ;
    weightHN;
    biasIR;
    biasIZ;
    biasIN;
    biasHR;
    biasHZ;
    biasHN;
    constructor(inputSize, hiddenSize, bias = true, device, dtype) {
        const bound = 1 / Math.sqrt(hiddenSize);
        this.weightIR = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightIZ = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightIN = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHR = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHZ = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHN = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        if (bias) {
            this.biasIR = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasIZ = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasIN = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHR = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHZ = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHN = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        }
    }
    forward(input, hidden) {
        input = this.weightIN.handleOther(input);
        hidden = this.weightHN.handleOther(hidden);
        const r = rnnTransform(input, hidden, this.weightIR, this.weightHR, this.biasIR, this.biasHR).sigmoid();
        const z = rnnTransform(input, hidden, this.weightIZ, this.weightHZ, this.biasIZ, this.biasHZ).sigmoid();
        const n = linearTransform(input, this.weightIN, this.biasIN).add(r.mul(linearTransform(hidden, this.weightHN, this.biasHN))).tanh();
        return (z.neg().add(1).mul(n).add(z.mul(hidden)));
    }
}
exports.GRUCell = GRUCell;
class LSTMCell {
    weightII;
    weightIF;
    weightIG;
    weightIO;
    weightHI;
    weightHF;
    weightHG;
    weightHO;
    biasII;
    biasIF;
    biasIG;
    biasIO;
    biasHI;
    biasHF;
    biasHG;
    biasHO;
    constructor(inputSize, hiddenSize, bias = true, device, dtype) {
        const bound = 1 / Math.sqrt(hiddenSize);
        this.weightII = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightIF = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightIG = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightIO = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHI = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHF = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHG = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHO = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        if (bias) {
            this.biasII = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasIF = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasIG = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasIO = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHI = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHF = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHG = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHO = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        }
    }
    forward(input, hidden, cell) {
        input = this.weightII.handleOther(input);
        hidden = this.weightHI.handleOther(hidden);
        cell = this.weightHI.handleOther(cell);
        const i = rnnTransform(input, hidden, this.weightII, this.weightHI, this.biasII, this.biasHI).sigmoid();
        const f = rnnTransform(input, hidden, this.weightIF, this.weightHF, this.biasIF, this.biasHF).sigmoid();
        const g = rnnTransform(input, hidden, this.weightIG, this.weightHG, this.biasIG, this.biasHG).tanh();
        const o = rnnTransform(input, hidden, this.weightIO, this.weightHO, this.biasIO, this.biasHO).sigmoid();
        const c = f.mul(cell).add(i.mul(g));
        const h = o.mul(c.tanh());
        return [h, c];
    }
}
exports.LSTMCell = LSTMCell;
class BatchNorm {
    weight;
    bias;
    runningMean;
    runningVar;
    eps;
    momentum;
    numFeatures;
    affine;
    trackRunningStats;
    numBatchesTracked;
    constructor(numFeatures, eps = 1e-5, momentum = 0.1, affine = true, trackRunningStats = true, device, dtype) {
        this.numFeatures = numFeatures;
        this.eps = eps;
        this.momentum = momentum;
        this.affine = affine;
        this.trackRunningStats = trackRunningStats;
        this.numBatchesTracked = 0;
        if (this.affine) {
            this.weight = core_1.Tensor.ones([numFeatures], { requiresGrad: true, device, dtype });
            this.bias = core_1.Tensor.zeros([numFeatures], { requiresGrad: true, device, dtype });
        }
        if (this.trackRunningStats) {
            this.runningMean = core_1.Tensor.zeros([numFeatures], { requiresGrad: false, device, dtype });
            this.runningVar = core_1.Tensor.ones([numFeatures], { requiresGrad: false, device, dtype });
        }
    }
    forward(input) {
        // Input shape: (N, C, ...) where C = numFeatures
        // Normalize over batch dimension and spatial dimensions (if any)
        if (input.shape.length < 2) {
            throw new Error("Input must have at least 2 dimensions (batch, features)");
        }
        if (input.shape[1] !== this.numFeatures) {
            throw new Error(`Expected ${this.numFeatures} features, got ${input.shape[1]}`);
        }
        let mean;
        let variance;
        if (core_1.Tensor.training || !this.trackRunningStats) {
            // Training or trackRunningStats disabled - calculate mean and variance from scratch
            // Calculate mean and variance over batch and spatial dimensions
            // Keep only the channel dimension
            const dims = [0, ...Array.from({ length: input.shape.length - 2 }, (_, i) => i + 2)];
            mean = input.mean(dims, true);
            variance = input.sub(mean).pow(2).mean(dims, true);
            // Update running statistics if enabled and in training mode
            if (this.trackRunningStats && core_1.Tensor.training) {
                const exponentialAverageFactor = this.momentum;
                this.runningMean = this.runningMean
                    .mul(1 - exponentialAverageFactor)
                    .add(mean.squeeze().mul(exponentialAverageFactor));
                // Use unbiased variance for running estimate
                const n = input.shape.reduce((acc, val, idx) => idx === 1 ? acc : acc * val, 1);
                const unbiasingFactor = n / (n - 1);
                this.runningVar = this.runningVar
                    .mul(1 - exponentialAverageFactor)
                    .add(variance.squeeze().mul(exponentialAverageFactor * unbiasingFactor));
                this.numBatchesTracked++;
            }
        }
        else {
            // Inference with trackRunningStats enabled - use running statistics
            mean = this.runningMean.reshape([1, this.numFeatures, ...Array(input.shape.length - 2).fill(1)]);
            variance = this.runningVar.reshape([1, this.numFeatures, ...Array(input.shape.length - 2).fill(1)]);
        }
        // Normalize
        let normalized = input.sub(mean).div(variance.add(this.eps).sqrt());
        // Apply affine transformation
        if (this.affine) {
            const weightReshaped = this.weight.reshape([1, this.numFeatures, ...Array(input.shape.length - 2).fill(1)]);
            const biasReshaped = this.bias.reshape([1, this.numFeatures, ...Array(input.shape.length - 2).fill(1)]);
            normalized = normalized.mul(weightReshaped).add(biasReshaped);
        }
        return normalized;
    }
}
exports.BatchNorm = BatchNorm;
class InstanceNorm {
    weight;
    bias;
    eps;
    numFeatures;
    constructor(numFeatures, eps = 1e-5, affine = true, device, dtype) {
        this.numFeatures = numFeatures;
        this.eps = eps;
        if (affine) {
            this.weight = core_1.Tensor.ones([numFeatures], { requiresGrad: true, device, dtype });
            this.bias = core_1.Tensor.zeros([numFeatures], { requiresGrad: true, device, dtype });
        }
    }
    forward(input) {
        // Input should be at least 3D: [N, C, ...spatial dims]
        if (input.shape.length < 3) {
            throw new Error("InstanceNorm expects at least 3D input [N, C, ...spatial]");
        }
        if (input.shape[1] !== this.numFeatures) {
            throw new Error(`Expected ${this.numFeatures} channels, got ${input.shape[1]}`);
        }
        // Normalize across spatial dimensions (all dims after channel dim)
        const dims = [];
        for (let i = 2; i < input.shape.length; i++) {
            dims.push(i);
        }
        const mean = input.mean(dims, true);
        const variance = input.sub(mean).pow(2).mean(dims, true);
        let normalized = input.sub(mean).div(variance.add(this.eps).sqrt());
        if (this.weight) {
            // Reshape weight to [1, C, 1, 1, ...] for broadcasting
            const weightShape = [1, this.numFeatures, ...Array(input.shape.length - 2).fill(1)];
            const weightReshaped = this.weight.reshape(weightShape);
            normalized = normalized.mul(weightReshaped);
        }
        if (this.bias) {
            // Reshape bias to [1, C, 1, 1, ...] for broadcasting
            const biasShape = [1, this.numFeatures, ...Array(input.shape.length - 2).fill(1)];
            const biasReshaped = this.bias.reshape(biasShape);
            normalized = normalized.add(biasReshaped);
        }
        return normalized;
    }
}
exports.InstanceNorm = InstanceNorm;
class GroupNorm {
    weight;
    bias;
    eps;
    numGroups;
    numChannels;
    constructor(numGroups, numChannels, eps = 1e-5, affine = true, device, dtype) {
        if (numChannels % numGroups !== 0) {
            throw new Error(`num_channels (${numChannels}) must be divisible by num_groups (${numGroups})`);
        }
        this.numGroups = numGroups;
        this.numChannels = numChannels;
        this.eps = eps;
        if (affine) {
            this.weight = core_1.Tensor.ones([numChannels], { requiresGrad: true, device, dtype });
            this.bias = core_1.Tensor.zeros([numChannels], { requiresGrad: true, device, dtype });
        }
    }
    forward(input) {
        // Input should be at least 3D: [N, C, ...spatial dims]
        if (input.shape.length < 3) {
            throw new Error("GroupNorm expects at least 3D input [N, C, ...spatial]");
        }
        if (input.shape[1] !== this.numChannels) {
            throw new Error(`Expected ${this.numChannels} channels, got ${input.shape[1]}`);
        }
        const N = input.shape[0];
        const C = input.shape[1];
        const spatialDims = input.shape.slice(2);
        const channelsPerGroup = C / this.numGroups;
        // Reshape: [N, C, ...spatial] -> [N, G, C//G, ...spatial]
        const reshapedInput = input.reshape([N, this.numGroups, channelsPerGroup, ...spatialDims]);
        // Normalize across (C//G, ...spatial) dimensions for each group
        // That's dims [2, 3, 4, ...] in the reshaped tensor
        const dims = [];
        for (let i = 2; i < reshapedInput.shape.length; i++) {
            dims.push(i);
        }
        const mean = reshapedInput.mean(dims, true);
        const variance = reshapedInput.sub(mean).pow(2).mean(dims, true);
        let normalized = reshapedInput.sub(mean).div(variance.add(this.eps).sqrt());
        // Reshape back: [N, G, C//G, ...spatial] -> [N, C, ...spatial]
        normalized = normalized.reshape(input.shape);
        if (this.weight) {
            // Reshape weight to [1, C, 1, 1, ...] for broadcasting
            const weightShape = [1, this.numChannels, ...Array(spatialDims.length).fill(1)];
            const weightReshaped = this.weight.reshape(weightShape);
            normalized = normalized.mul(weightReshaped);
        }
        if (this.bias) {
            // Reshape bias to [1, C, 1, 1, ...] for broadcasting
            const biasShape = [1, this.numChannels, ...Array(spatialDims.length).fill(1)];
            const biasReshaped = this.bias.reshape(biasShape);
            normalized = normalized.add(biasReshaped);
        }
        return normalized;
    }
}
exports.GroupNorm = GroupNorm;
class LayerNorm {
    weight;
    bias;
    eps;
    normalizedShape;
    constructor(normalizedShape, eps = 1e-5, elementwiseAffine = true, bias = true, device, dtype) {
        this.eps = eps;
        this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];
        if (this.normalizedShape.length === 0) {
            throw new Error("Normalized shape cannot be empty");
        }
        if (elementwiseAffine) {
            this.weight = core_1.Tensor.ones(this.normalizedShape, { requiresGrad: true, device, dtype });
            if (bias) {
                this.bias = core_1.Tensor.zeros(this.normalizedShape, { requiresGrad: true, device, dtype });
            }
        }
    }
    forward(input) {
        // Normalize over the specified dimensions
        const normalizedDims = this.normalizedShape.length;
        const startDim = input.shape.length - normalizedDims;
        if (startDim < 0) {
            throw new Error("Input does not have enough dims to normalize");
        }
        const dims = [];
        for (let i = 0; i < normalizedDims; i++) {
            if (input.shape[startDim + i] !== this.normalizedShape[i]) {
                throw new Error(`Shape mismatch at dim ${startDim + i}: expected ${this.normalizedShape[i]}, got ${input.shape[startDim + i]}`);
            }
            dims.push(startDim + i);
        }
        const mean = input.mean(dims, true);
        const variance = input.sub(mean).pow(2).mean(dims, true);
        let normalized = input.sub(mean).div(variance.add(this.eps).sqrt());
        if (this.weight) {
            normalized = normalized.mul(this.weight);
        }
        if (this.bias) {
            normalized = normalized.add(this.bias);
        }
        return normalized;
    }
}
exports.LayerNorm = LayerNorm;
class RMSNorm {
    weight;
    eps;
    normalizedShape;
    constructor(normalizedShape, eps = 1e-5, elementwiseAffine = true, device, dtype) {
        this.eps = eps;
        this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];
        if (this.normalizedShape.length === 0) {
            throw new Error("Normalized shape cannot be empty");
        }
        if (elementwiseAffine) {
            this.weight = core_1.Tensor.ones(this.normalizedShape, { requiresGrad: true, device, dtype });
        }
    }
    forward(input) {
        // Normalize over the specified dimensions
        const normalizedDims = this.normalizedShape.length;
        const startDim = input.shape.length - normalizedDims;
        if (startDim < 0) {
            throw new Error("Input does not have enough dims to normalize");
        }
        const dims = [];
        for (let i = 0; i < normalizedDims; i++) {
            if (input.shape[startDim + i] !== this.normalizedShape[i]) {
                throw new Error(`Shape mismatch at dim ${startDim + i}: expected ${this.normalizedShape[i]}, got ${input.shape[startDim + i]}`);
            }
            dims.push(startDim + i);
        }
        let rms = input.square().mean(dims, true).add(this.eps).sqrt();
        let normalized = input.div(rms);
        if (this.weight) {
            normalized = normalized.mul(this.weight);
        }
        return normalized;
    }
}
exports.RMSNorm = RMSNorm;
class Embedding {
    weight;
    constructor(numEmbeddings, embeddingDim, device, dtype) {
        this.weight = core_1.Tensor.randn([numEmbeddings, embeddingDim], { requiresGrad: true, device, dtype });
    }
    forward(input) {
        return this.weight.index(input);
    }
}
exports.Embedding = Embedding;
function scaledDotProductAttention(query, key, value, attnMask, dropout = 0, isCausal = false, scale) {
    const targetLen = query.shape[query.shape.length - 2];
    const sourceLen = key.shape[key.shape.length - 2];
    const dimSize = query.shape[query.shape.length - 1];
    // Attention scores
    let scores = query.matmul(key.transpose(-2, -1)).div(scale ?? Math.sqrt(dimSize));
    // Set attention mask to causal mask if specified
    if (isCausal) {
        attnMask = core_1.Tensor.ones([targetLen, sourceLen], { device: query.device }).triu(1);
    }
    // Apply attention mask if specified
    if (attnMask) {
        scores = scores.maskedFill(attnMask, -Infinity);
    }
    // Calculate attention weights
    let attnWeights = scores.softmax().dropout(dropout);
    // Apply attention to values
    return attnWeights.matmul(value);
}
class MultiheadAttention {
    qProjection;
    kProjection;
    vProjection;
    oProjection;
    embedDim;
    numHeads;
    headDim;
    dropout;
    constructor(embedDim, numHeads, dropout = 0, bias = true, device, dtype) {
        this.qProjection = new Linear(embedDim, embedDim, bias, device, dtype);
        this.kProjection = new Linear(embedDim, embedDim, bias, device, dtype);
        this.vProjection = new Linear(embedDim, embedDim, bias, device, dtype);
        this.oProjection = new Linear(embedDim, embedDim, bias, device, dtype);
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = Math.floor(embedDim / numHeads);
        this.dropout = dropout;
    }
    forward(query, key, value, needWeights = true, attnMask, averageAttnWeights = true, isCausal = false) {
        // Batch-first
        const [batchSize, targetLen, embedDim] = query.shape;
        const sourceLen = key.shape[1];
        let Q = this.qProjection.forward(query); // (batchSize, targetLen, embedDim)
        let K = this.kProjection.forward(key); // (batchSize, sourceLen, embedDim)
        let V = this.vProjection.forward(value); // (batchSize, sourceLen, embedDim)
        // (batchSize, numHeads, targetLen/sourceLen, headDim)
        Q = Q.reshape([batchSize, targetLen, this.numHeads, this.headDim]).transpose(1, 2);
        K = K.reshape([batchSize, sourceLen, this.numHeads, this.headDim]).transpose(1, 2);
        V = V.reshape([batchSize, sourceLen, this.numHeads, this.headDim]).transpose(1, 2);
        // Attention scores
        let scores = Q.matmul(K.transpose(-2, -1)).div(Math.sqrt(this.headDim));
        // Set attention mask to causal mask if specified
        if (isCausal) {
            attnMask = core_1.Tensor.ones([targetLen, sourceLen], { device: this.qProjection.weight.device }).triu(1);
        }
        // Apply attention mask if specified
        if (attnMask) {
            scores = scores.maskedFill(attnMask, -Infinity);
        }
        // Calculate attention weights
        let attnWeights = scores.softmax().dropout(this.dropout);
        // Apply attention to values
        let attnOutput = attnWeights.matmul(V); // (batchSize, numHeads, targetLen, headDim)
        // (batchSize, targetLen, embedDim)
        attnOutput = attnOutput.transpose(1, 2).reshape([batchSize, targetLen, embedDim]);
        // Output
        const output = this.oProjection.forward(attnOutput);
        // Average weights if needed
        if (averageAttnWeights) {
            attnWeights = attnWeights.mean(1);
        }
        return [output, needWeights ? attnWeights : undefined];
    }
}
exports.MultiheadAttention = MultiheadAttention;
const state = {
    getParameters(model, visited = new WeakSet()) {
        if (visited.has(model))
            return [];
        visited.add(model);
        const parameters = [];
        for (const key in model) {
            if (!model.hasOwnProperty(key))
                continue;
            const value = model[key];
            if (value instanceof core_1.Tensor) {
                parameters.push(value);
            }
            else if (typeof value === "object" && value !== null) {
                parameters.push(...state.getParameters(value, visited));
            }
        }
        return parameters;
    },
    moveParameters(model, device) {
        const params = state.getParameters(model);
        for (const param of params) {
            param.to_(device);
        }
    },
    getStateDict(model, prefix = "", visited = new WeakSet()) {
        if (visited.has(model))
            return {};
        visited.add(model);
        const stateDict = {};
        for (const key in model) {
            if (!model.hasOwnProperty(key))
                continue;
            const value = model[key];
            const fullKey = prefix ? `${prefix}.${key}` : key;
            if (value instanceof core_1.Tensor) {
                stateDict[fullKey] = value.val();
            }
            else if (typeof value === "object" && value !== null) {
                Object.assign(stateDict, state.getStateDict(value, fullKey, visited));
            }
        }
        return stateDict;
    },
    loadStateDict(model, stateDict, prefix = "", visited = new WeakSet()) {
        if (visited.has(model))
            return;
        visited.add(model);
        for (const key in model) {
            if (!model.hasOwnProperty(key))
                continue;
            const value = model[key];
            const fullKey = prefix ? `${prefix}.${key}` : key;
            if (value instanceof core_1.Tensor && stateDict[fullKey]) {
                value.replace(new core_1.Tensor(stateDict[fullKey], { device: value.device }));
            }
            else if (typeof value === "object" && value !== null) {
                state.loadStateDict(value, stateDict, fullKey, visited);
            }
        }
    }
};
exports.nn = {
    Linear,
    RNNCell,
    GRUCell,
    LSTMCell,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
    LayerNorm,
    RMSNorm,
    Embedding,
    scaledDotProductAttention,
    MultiheadAttention,
    state
};

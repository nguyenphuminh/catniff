"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.nn = exports.MultiheadAttention = exports.Embedding = exports.RMSNorm = exports.LayerNorm = exports.LSTMCell = exports.GRUCell = exports.RNNCell = exports.Linear = void 0;
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
    constructor(inFeatures, outFeatures, bias = true, device) {
        const bound = 1 / Math.sqrt(inFeatures);
        this.weight = core_1.Tensor.uniform([outFeatures, inFeatures], -bound, bound, { requiresGrad: true, device });
        if (bias) {
            this.bias = core_1.Tensor.uniform([outFeatures], -bound, bound, { requiresGrad: true, device });
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
    constructor(inputSize, hiddenSize, bias = true, device) {
        const bound = 1 / Math.sqrt(hiddenSize);
        this.weightIH = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightHH = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });
        if (bias) {
            this.biasIH = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHH = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
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
    constructor(inputSize, hiddenSize, bias = true, device) {
        const bound = 1 / Math.sqrt(hiddenSize);
        this.weightIR = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightIZ = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightIN = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightHR = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });
        this.weightHZ = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });
        this.weightHN = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });
        if (bias) {
            this.biasIR = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasIZ = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasIN = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHR = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHZ = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHN = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
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
    constructor(inputSize, hiddenSize, bias = true, device) {
        const bound = 1 / Math.sqrt(hiddenSize);
        this.weightII = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightIF = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightIG = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightIO = core_1.Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightHI = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });
        this.weightHF = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });
        this.weightHG = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });
        this.weightHO = core_1.Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });
        if (bias) {
            this.biasII = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasIF = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasIG = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasIO = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHI = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHF = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHG = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHO = core_1.Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
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
class LayerNorm {
    weight;
    bias;
    eps;
    normalizedShape;
    constructor(normalizedShape, eps = 1e-5, elementwiseAffine = true, bias = true, device) {
        this.eps = eps;
        this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];
        if (this.normalizedShape.length === 0) {
            throw new Error("Normalized shape cannot be empty");
        }
        if (elementwiseAffine) {
            this.weight = core_1.Tensor.ones(this.normalizedShape, { requiresGrad: true, device });
            if (bias) {
                this.bias = core_1.Tensor.zeros(this.normalizedShape, { requiresGrad: true, device });
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
    constructor(normalizedShape, eps = 1e-5, elementwiseAffine = true, device) {
        this.eps = eps;
        this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];
        if (this.normalizedShape.length === 0) {
            throw new Error("Normalized shape cannot be empty");
        }
        if (elementwiseAffine) {
            this.weight = core_1.Tensor.ones(this.normalizedShape, { requiresGrad: true, device });
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
    constructor(numEmbeddings, embeddingDim, device) {
        this.weight = core_1.Tensor.randn([numEmbeddings, embeddingDim], { requiresGrad: true, device });
    }
    forward(input) {
        return this.weight.index(input);
    }
}
exports.Embedding = Embedding;
class MultiheadAttention {
    qProjection;
    kProjection;
    vProjection;
    oProjection;
    embedDim;
    numHeads;
    headDim;
    dropout;
    constructor(embedDim, numHeads, dropout = 0, bias = true, device) {
        this.qProjection = new Linear(embedDim, embedDim, bias, device);
        this.kProjection = new Linear(embedDim, embedDim, bias, device);
        this.vProjection = new Linear(embedDim, embedDim, bias, device);
        this.oProjection = new Linear(embedDim, embedDim, bias, device);
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = Math.floor(embedDim / numHeads);
        this.dropout = dropout;
    }
    forward(query, key, value, needWeights = true, attnMask, averageAttnWeights = true) {
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
    LayerNorm,
    RMSNorm,
    Embedding,
    MultiheadAttention,
    state
};

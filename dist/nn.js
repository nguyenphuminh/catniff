"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.nn = exports.LSTMCell = void 0;
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
        input = core_1.Tensor.forceTensor(input);
        return linearTransform(input, this.weight, this.bias);
    }
}
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
        input = core_1.Tensor.forceTensor(input);
        hidden = core_1.Tensor.forceTensor(hidden);
        return rnnTransform(input, hidden, this.weightIH, this.weightHH, this.biasIH, this.biasHH).tanh();
    }
}
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
        input = core_1.Tensor.forceTensor(input);
        hidden = core_1.Tensor.forceTensor(hidden);
        const r = rnnTransform(input, hidden, this.weightIR, this.weightHR, this.biasIR, this.biasHR).sigmoid();
        const z = rnnTransform(input, hidden, this.weightIZ, this.weightHZ, this.biasIZ, this.biasHZ).sigmoid();
        const n = linearTransform(input, this.weightIN, this.biasIN).add(r.mul(linearTransform(hidden, this.weightHN, this.biasHN))).tanh();
        return (z.neg().add(1).mul(n).add(z.mul(hidden)));
    }
}
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
        input = core_1.Tensor.forceTensor(input);
        hidden = core_1.Tensor.forceTensor(hidden);
        cell = core_1.Tensor.forceTensor(cell);
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
                Object.assign(stateDict, this.getStateDict(value, fullKey, visited));
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
                this.loadStateDict(value, stateDict, fullKey, visited);
            }
        }
    }
};
exports.nn = {
    Linear,
    RNNCell,
    GRUCell,
    state
};

"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.nn = void 0;
const core_1 = require("./core");
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
        let output = input.matmul(this.weight.t());
        if (this.bias) {
            output = output.add(this.bias);
        }
        return output;
    }
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
        let output = input.matmul(this.weightIH.t())
            .add(hidden.matmul(this.weightHH.t()));
        if (this.biasIH && this.biasHH) {
            output = output.add(this.biasIH).add(this.biasHH);
        }
        return output.tanh();
    }
}
const state = {
    getParameters(model, visited = new WeakSet()) {
        if (visited.has(model)) {
            return [];
        }
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
    }
};
exports.nn = {
    Linear,
    RNNCell,
    state
};

"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.nn = void 0;
const core_1 = require("./core");
class Linear {
    weight;
    bias;
    constructor(inFeatures, outFeatures, bias = true, customInit) {
        let initFunc = (shape) => {
            const bound = 1 / Math.sqrt(inFeatures);
            return core_1.Tensor.uniform(shape, -bound, bound, { requiresGrad: true });
        };
        if (customInit) {
            initFunc = customInit;
        }
        this.weight = initFunc([outFeatures, inFeatures]);
        if (bias) {
            this.bias = initFunc([outFeatures]);
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
    state
};

import { Tensor, TensorValue } from "./core";

class Linear {
    public weight: Tensor;
    public bias?: Tensor;

    constructor(
        inFeatures: number,
        outFeatures: number,
        bias: boolean = true,
        device?: string
    ) {
        const bound = 1 / Math.sqrt(inFeatures);
        this.weight = Tensor.uniform([outFeatures, inFeatures], -bound, bound, { requiresGrad: true, device });

        if (bias) {
            this.bias = Tensor.uniform([outFeatures], -bound, bound, { requiresGrad: true, device });
        }
    }

    forward(input: Tensor | TensorValue): Tensor {
        input = Tensor.forceTensor(input);

        let output = input.matmul(this.weight.t());

        if (this.bias) {
            output = output.add(this.bias);
        }

        return output;
    }
}

class RNNCell {
    public weightIH: Tensor;
    public weightHH: Tensor;
    public biasIH?: Tensor;
    public biasHH?: Tensor;

    constructor(
        inputSize: number,
        hiddenSize: number,
        bias: boolean = true,
        device?: string
    ) {
        const bound = 1 / Math.sqrt(hiddenSize);
        this.weightIH = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightHH = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });

        if (bias) {
            this.biasIH = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHH = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
        }
    }

    forward(input: Tensor | TensorValue, hidden: Tensor | TensorValue): Tensor {
        input = Tensor.forceTensor(input);
        hidden = Tensor.forceTensor(hidden);

        let output = input.matmul(this.weightIH.t())
            .add(hidden.matmul(this.weightHH.t()));

        if (this.biasIH && this.biasHH) {
            output = output.add(this.biasIH).add(this.biasHH);
        }

        return output.tanh();
    }
}

const state = {
    getParameters(model: any, visited: WeakSet<object> = new WeakSet()) {
        if (visited.has(model)) {
            return [];
        }

        visited.add(model);

        const parameters: Tensor[] = [];

        for (const key in model) {
            if (!model.hasOwnProperty(key)) continue;

            const value = model[key];

            if (value instanceof Tensor) {
                parameters.push(value);
            } else if (typeof value === "object" && value !== null) {
                parameters.push(...state.getParameters(value, visited));
            }
        }

        return parameters;
    }
}

export const nn = {
    Linear,
    RNNCell,
    state
}

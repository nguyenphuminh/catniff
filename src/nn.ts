import { Tensor, TensorValue } from "./core";

class Linear {
    public weight: Tensor;
    public bias?: Tensor;

    constructor(
        inFeatures: number,
        outFeatures: number,
        bias: boolean = true,
        customInit?: (shape: number[]) => Tensor
    ) {
        let initFunc = (shape: number[]) => {
            const bound = 1 / Math.sqrt(inFeatures);

            return Tensor.uniform(shape, -bound, bound, { requiresGrad: true });
        }

        if (customInit) {
            initFunc = customInit;
        }

        this.weight = initFunc([outFeatures, inFeatures]);

        if (bias) {
            this.bias = initFunc([outFeatures]);
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
    state
}

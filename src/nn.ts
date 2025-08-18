import { Tensor, TensorValue } from "./core";

function linearTransform(input: Tensor, weight: Tensor, bias?: Tensor): Tensor {
    let output = input.matmul(weight.t());

    if (bias) {
        output = output.add(bias);
    }

    return output;
}

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

        return linearTransform(input, this.weight, this.bias);
    }
}

function rnnTransform(
    input: Tensor,
    hidden: Tensor,
    inputWeight: Tensor,
    hiddenWeight: Tensor,
    inputBias?: Tensor,
    hiddenBias?: Tensor
): Tensor {
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

        return rnnTransform(input, hidden, this.weightIH, this.weightHH, this.biasIH, this.biasHH).tanh();
    }
}

class GRUCell {
    public weightIR: Tensor;
    public weightIZ: Tensor;
    public weightIN: Tensor;
    public weightHR: Tensor;
    public weightHZ: Tensor;
    public weightHN: Tensor;
    public biasIR?: Tensor;
    public biasIZ?: Tensor;
    public biasIN?: Tensor;
    public biasHR?: Tensor;
    public biasHZ?: Tensor;
    public biasHN?: Tensor;

    constructor(
        inputSize: number,
        hiddenSize: number,
        bias: boolean = true,
        device?: string
    ) {
        const bound = 1 / Math.sqrt(hiddenSize);
        this.weightIR = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightIZ = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightIN = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightHR = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });
        this.weightHZ = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });
        this.weightHN = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });

        if (bias) {
            this.biasIR = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasIZ = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasIN = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHR = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHZ = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHN = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
        }
    }

    forward(input: Tensor | TensorValue, hidden: Tensor | TensorValue): Tensor {
        input = Tensor.forceTensor(input);
        hidden = Tensor.forceTensor(hidden);

        const r = rnnTransform(input, hidden, this.weightIR, this.weightHR, this.biasIR, this.biasHR).sigmoid();
        const z = rnnTransform(input, hidden, this.weightIZ, this.weightHZ, this.biasIZ, this.biasHZ).sigmoid();
        const n = linearTransform(input, this.weightIN, this.biasIN).add(r.mul(linearTransform(hidden, this.weightHN, this.biasHN))).tanh();

        return (z.neg().add(1).mul(n).add(z.mul(hidden)));
    }
}

export class LSTMCell {
    public weightII: Tensor;
    public weightIF: Tensor;
    public weightIG: Tensor;
    public weightIO: Tensor;
    public weightHI: Tensor;
    public weightHF: Tensor;
    public weightHG: Tensor;
    public weightHO: Tensor;
    public biasII?: Tensor;
    public biasIF?: Tensor;
    public biasIG?: Tensor;
    public biasIO?: Tensor;
    public biasHI?: Tensor;
    public biasHF?: Tensor;
    public biasHG?: Tensor;
    public biasHO?: Tensor;

    constructor(
        inputSize: number,
        hiddenSize: number,
        bias: boolean = true,
        device?: string
    ) {
        const bound = 1 / Math.sqrt(hiddenSize);
        this.weightII = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightIF = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightIG = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightIO = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device });
        this.weightHI = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });
        this.weightHF = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });
        this.weightHG = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });
        this.weightHO = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device });

        if (bias) {
            this.biasII = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasIF = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasIG = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasIO = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHI = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHF = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHG = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
            this.biasHO = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device });
        }
    }

    forward(
        input: Tensor | TensorValue,
        hidden: Tensor | TensorValue,
        cell: Tensor | TensorValue
    ): [Tensor, Tensor] {
        input = Tensor.forceTensor(input);
        hidden = Tensor.forceTensor(hidden);
        cell = Tensor.forceTensor(cell);

        const i = rnnTransform(input, hidden, this.weightII, this.weightHI, this.biasII, this.biasHI).sigmoid();
        const f = rnnTransform(input, hidden, this.weightIF, this.weightHF, this.biasIF, this.biasHF).sigmoid();
        const g = rnnTransform(input, hidden, this.weightIG, this.weightHG, this.biasIG, this.biasHG).tanh();
        const o = rnnTransform(input, hidden, this.weightIO, this.weightHO, this.biasIO, this.biasHO).sigmoid();
        const c = f.mul(cell).add(i.mul(g));
        const h = o.mul(c.tanh());

        return [h, c];
    }
}

interface StateDict {
    [key: string]: any; // Could be nested objects or tensor data
}

const state = {
    getParameters(model: any, visited: WeakSet<object> = new WeakSet()) {
        if (visited.has(model)) return [];

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
    },

    getStateDict(model: any, prefix: string = "", visited: WeakSet<object> = new WeakSet()): StateDict {
        if (visited.has(model)) return {};

        visited.add(model);

        const stateDict: StateDict = {};

        for (const key in model) {
            if (!model.hasOwnProperty(key)) continue;

            const value = model[key];
            const fullKey = prefix ? `${prefix}.${key}` : key;

            if (value instanceof Tensor) {
                stateDict[fullKey] = value.val();
            } else if (typeof value === "object" && value !== null) {
                Object.assign(stateDict, this.getStateDict(value, fullKey, visited));
            }
        }

        return stateDict;
    },

    loadStateDict(model: any, stateDict: StateDict, prefix: string = "", visited: WeakSet<object> = new WeakSet()): void {
        if (visited.has(model)) return;

        visited.add(model);

        for (const key in model) {
            if (!model.hasOwnProperty(key)) continue;

            const value = model[key];
            const fullKey = prefix ? `${prefix}.${key}` : key;

            if (value instanceof Tensor && stateDict[fullKey]) {
                value.replace(new Tensor(stateDict[fullKey], { device: value.device }));
            } else if (typeof value === "object" && value !== null) {
                this.loadStateDict(value, stateDict, fullKey, visited);
            }
        }
    }
}

export const nn = {
    Linear,
    RNNCell,
    GRUCell,
    state
}

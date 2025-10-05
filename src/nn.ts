import { Tensor, TensorValue } from "./core";

function linearTransform(input: Tensor, weight: Tensor, bias?: Tensor): Tensor {
    let output = input.matmul(weight.t());

    if (bias) {
        output = output.add(bias);
    }

    return output;
}

export class Linear {
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
        input = this.weight.handleOther(input);

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

export class RNNCell {
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
        input = this.weightIH.handleOther(input);
        hidden = this.weightHH.handleOther(hidden);

        return rnnTransform(input, hidden, this.weightIH, this.weightHH, this.biasIH, this.biasHH).tanh();
    }
}

export class GRUCell {
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
        input = this.weightIN.handleOther(input);
        hidden = this.weightHN.handleOther(hidden);

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

export class LayerNorm {
    public weight?: Tensor;
    public bias?: Tensor;
    public eps: number;
    public normalizedShape: number[];

    constructor(
        normalizedShape: number | number[],
        eps: number = 1e-5,
        elementwiseAffine: boolean = true,
        bias: boolean = true,
        device?: string
    ) {
        this.eps = eps;
        this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];

        if (this.normalizedShape.length === 0) {
            throw new Error("Normalized shape cannot be empty");
        }

        if (elementwiseAffine) {
            this.weight = Tensor.ones(this.normalizedShape, { requiresGrad: true, device });

            if (bias) {
                this.bias = Tensor.zeros(this.normalizedShape, { requiresGrad: true, device });
            }
        }
    }

    forward(input: Tensor): Tensor {
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

export class RMSNorm {
    public weight?: Tensor;
    public eps: number;
    public normalizedShape: number[];

    constructor(
        normalizedShape: number | number[],
        eps: number = 1e-5,
        elementwiseAffine: boolean = true,
        device?: string
    ) {
        this.eps = eps;
        this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];

        if (this.normalizedShape.length === 0) {
            throw new Error("Normalized shape cannot be empty");
        }

        if (elementwiseAffine) {
            this.weight = Tensor.ones(this.normalizedShape, { requiresGrad: true, device });
        }
    }

    forward(input: Tensor): Tensor {
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

export class Embedding {
    public weight: Tensor;

    constructor(numEmbeddings: number, embeddingDim: number, device: string) {
        this.weight = Tensor.randn([numEmbeddings, embeddingDim], { requiresGrad: true, device });
    }

    forward(input: Tensor | TensorValue): Tensor {
        return this.weight.index(input);
    }
}

export class MultiheadAttention {
    public qProjection: Linear;
    public kProjection: Linear;
    public vProjection: Linear;
    public oProjection: Linear;

    public embedDim: number;
    public numHeads: number;
    public headDim: number;
    public dropout: number;

    constructor(
        embedDim: number,
        numHeads: number,
        dropout = 0,
        bias = true,
        device?: string
    ) {
        this.qProjection = new Linear(embedDim, embedDim, bias, device);
        this.kProjection = new Linear(embedDim, embedDim, bias, device);
        this.vProjection = new Linear(embedDim, embedDim, bias, device);
        this.oProjection = new Linear(embedDim, embedDim, bias, device);

        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = Math.floor(embedDim / numHeads);
        this.dropout = dropout;
    }

    forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        needWeights = true,
        attnMask?: Tensor,
        averageAttnWeights = true
    ): [Tensor, Tensor | undefined] {
        // Batch-first
        const [batchSize, targetLen, embedDim] = query.shape;
        const sourceLen = key.shape[1];

        let Q = this.qProjection.forward(query); // (batchSize, targetLen, embedDim)
        let K = this.kProjection.forward(key);   // (batchSize, sourceLen, embedDim)
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

export interface StateDict {
    [key: string]: any; // Could be nested objects or tensor data
}

const state = {
    getParameters(model: any, visited: WeakSet<object> = new WeakSet()): Tensor[] {
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

    moveParameters(model: any, device: string): void {
        const params = state.getParameters(model);

        for (const param of params) {
            param.to_(device);
        }
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
                Object.assign(stateDict, state.getStateDict(value, fullKey, visited));
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
                state.loadStateDict(value, stateDict, fullKey, visited);
            }
        }
    }
}

export const nn = {
    Linear,
    RNNCell,
    GRUCell,
    LSTMCell,
    LayerNorm,
    RMSNorm,
    Embedding,
    MultiheadAttention,
    state
}

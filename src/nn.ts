import { Tensor, TensorValue } from "./core";
import { dtype } from "./dtype";

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
        device?: string,
        dtype?: dtype
    ) {
        const bound = 1 / Math.sqrt(inFeatures);
        this.weight = Tensor.uniform([outFeatures, inFeatures], -bound, bound, { requiresGrad: true, device, dtype });

        if (bias) {
            this.bias = Tensor.uniform([outFeatures], -bound, bound, { requiresGrad: true, device, dtype });
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
        device?: string,
        dtype?: dtype
    ) {
        const bound = 1 / Math.sqrt(hiddenSize);
        this.weightIH = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHH = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });

        if (bias) {
            this.biasIH = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHH = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
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
        device?: string,
        dtype?: dtype
    ) {
        const bound = 1 / Math.sqrt(hiddenSize);
        this.weightIR = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightIZ = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightIN = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHR = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHZ = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHN = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });

        if (bias) {
            this.biasIR = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasIZ = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasIN = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHR = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHZ = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHN = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
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
        device?: string,
        dtype?: dtype
    ) {
        const bound = 1 / Math.sqrt(hiddenSize);
        this.weightII = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightIF = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightIG = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightIO = Tensor.uniform([hiddenSize, inputSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHI = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHF = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHG = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
        this.weightHO = Tensor.uniform([hiddenSize, hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });

        if (bias) {
            this.biasII = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasIF = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasIG = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasIO = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHI = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHF = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHG = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
            this.biasHO = Tensor.uniform([hiddenSize], -bound, bound, { requiresGrad: true, device, dtype });
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

export class BatchNorm {
    public weight?: Tensor;
    public bias?: Tensor;
    public runningMean?: Tensor;
    public runningVar?: Tensor;
    public eps: number;
    public momentum: number;
    public numFeatures: number;
    public affine: boolean;
    public trackRunningStats: boolean;
    public numBatchesTracked: number;

    constructor(
        numFeatures: number,
        eps: number = 1e-5,
        momentum: number = 0.1,
        affine: boolean = true,
        trackRunningStats: boolean = true,
        device?: string,
        dtype?: dtype
    ) {
        this.numFeatures = numFeatures;
        this.eps = eps;
        this.momentum = momentum;
        this.affine = affine;
        this.trackRunningStats = trackRunningStats;
        this.numBatchesTracked = 0;

        if (this.affine) {
            this.weight = Tensor.ones([numFeatures], { requiresGrad: true, device, dtype });
            this.bias = Tensor.zeros([numFeatures], { requiresGrad: true, device, dtype });
        }

        if (this.trackRunningStats) {
            this.runningMean = Tensor.zeros([numFeatures], { requiresGrad: false, device, dtype });
            this.runningVar = Tensor.ones([numFeatures], { requiresGrad: false, device, dtype });
        }
    }

    forward(input: Tensor): Tensor {
        // Input shape: (N, C, ...) where C = numFeatures
        // Normalize over batch dimension and spatial dimensions (if any)

        if (input.shape.length < 2) {
            throw new Error("Input must have at least 2 dimensions (batch, features)");
        }

        if (input.shape[1] !== this.numFeatures) {
            throw new Error(`Expected ${this.numFeatures} features, got ${input.shape[1]}`);
        }

        let mean: Tensor;
        let variance: Tensor;

        if (Tensor.training || !this.trackRunningStats) {
            // Training or trackRunningStats disabled - calculate mean and variance from scratch

            // Calculate mean and variance over batch and spatial dimensions
            // Keep only the channel dimension
            const dims = [0, ...Array.from({ length: input.shape.length - 2 }, (_, i) => i + 2)];

            mean = input.mean(dims, true);
            variance = input.sub(mean).pow(2).mean(dims, true);

            // Update running statistics if enabled and in training mode
            if (this.trackRunningStats && Tensor.training) {
                const exponentialAverageFactor = this.momentum;

                this.runningMean = this.runningMean!
                    .mul(1 - exponentialAverageFactor)
                    .add(mean.squeeze().mul(exponentialAverageFactor));

                // Use unbiased variance for running estimate
                const n = input.shape.reduce((acc, val, idx) =>
                    idx === 1 ? acc : acc * val, 1
                );
                const unbiasingFactor = n / (n - 1);

                this.runningVar = this.runningVar!
                    .mul(1 - exponentialAverageFactor)
                    .add(variance.squeeze().mul(exponentialAverageFactor * unbiasingFactor));

                this.numBatchesTracked++;
            }
        } else {
            // Inference with trackRunningStats enabled - use running statistics
            mean = this.runningMean!.reshape([1, this.numFeatures, ...Array(input.shape.length - 2).fill(1)]);
            variance = this.runningVar!.reshape([1, this.numFeatures, ...Array(input.shape.length - 2).fill(1)]);
        }

        // Normalize
        let normalized = input.sub(mean).div(variance.add(this.eps).sqrt());

        // Apply affine transformation
        if (this.affine) {
            const weightReshaped = this.weight!.reshape([1, this.numFeatures, ...Array(input.shape.length - 2).fill(1)]);
            const biasReshaped = this.bias!.reshape([1, this.numFeatures, ...Array(input.shape.length - 2).fill(1)]);

            normalized = normalized.mul(weightReshaped).add(biasReshaped);
        }

        return normalized;
    }
}

export class InstanceNorm {
    public weight?: Tensor;
    public bias?: Tensor;
    public eps: number;
    public numFeatures: number;
    
    constructor(
        numFeatures: number,
        eps: number = 1e-5,
        affine: boolean = true,
        device?: string,
        dtype?: dtype
    ) {
        this.numFeatures = numFeatures;
        this.eps = eps;
        
        if (affine) {
            this.weight = Tensor.ones([numFeatures], { requiresGrad: true, device, dtype });
            this.bias = Tensor.zeros([numFeatures], { requiresGrad: true, device, dtype });
        }
    }
    
    forward(input: Tensor): Tensor {
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

export class GroupNorm {
    public weight?: Tensor;
    public bias?: Tensor;
    public eps: number;
    public numGroups: number;
    public numChannels: number;
    
    constructor(
        numGroups: number,
        numChannels: number,
        eps: number = 1e-5,
        affine: boolean = true,
        device?: string,
        dtype?: dtype
    ) {
        if (numChannels % numGroups !== 0) {
            throw new Error(`num_channels (${numChannels}) must be divisible by num_groups (${numGroups})`);
        }
        
        this.numGroups = numGroups;
        this.numChannels = numChannels;
        this.eps = eps;
        
        if (affine) {
            this.weight = Tensor.ones([numChannels], { requiresGrad: true, device, dtype });
            this.bias = Tensor.zeros([numChannels], { requiresGrad: true, device, dtype });
        }
    }
    
    forward(input: Tensor): Tensor {
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
        device?: string,
        dtype?: dtype
    ) {
        this.eps = eps;
        this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];

        if (this.normalizedShape.length === 0) {
            throw new Error("Normalized shape cannot be empty");
        }

        if (elementwiseAffine) {
            this.weight = Tensor.ones(this.normalizedShape, { requiresGrad: true, device, dtype });

            if (bias) {
                this.bias = Tensor.zeros(this.normalizedShape, { requiresGrad: true, device, dtype });
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
        device?: string,
        dtype?: dtype
    ) {
        this.eps = eps;
        this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];

        if (this.normalizedShape.length === 0) {
            throw new Error("Normalized shape cannot be empty");
        }

        if (elementwiseAffine) {
            this.weight = Tensor.ones(this.normalizedShape, { requiresGrad: true, device, dtype });
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

    constructor(numEmbeddings: number, embeddingDim: number, device?: string, dtype?: dtype) {
        this.weight = Tensor.randn([numEmbeddings, embeddingDim], { requiresGrad: true, device, dtype });
    }

    forward(input: Tensor | TensorValue): Tensor {
        return this.weight.index(input);
    }
}

export function scaledDotProductAttention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attnMask?: Tensor,
    dropout = 0,
    isCausal = false,
    scale?: number
) {
    const targetLen = query.shape[query.shape.length - 2];
    const sourceLen = key.shape[key.shape.length - 2];
    const dimSize = query.shape[query.shape.length - 1];

    // Attention scores
    let scores = query.matmul(key.transpose(-2, -1)).div(scale ?? Math.sqrt(dimSize));

    // Set attention mask to causal mask if specified
    if (isCausal) {
        attnMask = Tensor.ones([targetLen, sourceLen], { device: query.device }).triu(1);
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
        device?: string,
        dtype?: dtype
    ) {
        this.qProjection = new Linear(embedDim, embedDim, bias, device, dtype);
        this.kProjection = new Linear(embedDim, embedDim, bias, device, dtype);
        this.vProjection = new Linear(embedDim, embedDim, bias, device, dtype);
        this.oProjection = new Linear(embedDim, embedDim, bias, device, dtype);

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
        averageAttnWeights = true,
        isCausal = false
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

        // Set attention mask to causal mask if specified
        if (isCausal) {
            attnMask = Tensor.ones([targetLen, sourceLen], { device: this.qProjection.weight.device }).triu(1);
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
    BatchNorm,
    InstanceNorm,
    GroupNorm,
    LayerNorm,
    RMSNorm,
    Embedding,
    scaledDotProductAttention,
    MultiheadAttention,
    state
}

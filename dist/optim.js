"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Optim = exports.AdamW = exports.Adam = exports.SGD = exports.BaseOptimizer = void 0;
const core_1 = require("./core");
class BaseOptimizer {
    paramGroups;
    constructor(params) {
        if (params[0] instanceof core_1.Tensor) {
            this.paramGroups = [{ params: params }];
        }
        else {
            this.paramGroups = params;
        }
    }
    zeroGrad(del = true) {
        for (let index = 0; index < this.paramGroups.length; index++) {
            const paramGroup = this.paramGroups[index];
            for (const param of paramGroup.params) {
                if (del) {
                    delete param.grad;
                }
                else {
                    param.grad = core_1.Tensor.zerosLike(param);
                }
            }
        }
    }
}
exports.BaseOptimizer = BaseOptimizer;
class SGD extends BaseOptimizer {
    lr;
    momentum;
    dampening;
    weightDecay;
    nesterov;
    momentumBuffers = new Map();
    constructor(params, options) {
        super(params);
        this.lr = options?.lr ?? 0.001;
        this.momentum = options?.momentum ?? 0;
        this.dampening = options?.dampening ?? 0;
        this.weightDecay = options?.weightDecay ?? 0;
        this.nesterov = options?.nesterov ?? false;
    }
    step() {
        for (const paramGroup of this.paramGroups) {
            const lr = paramGroup.lr ?? this.lr;
            const momentum = paramGroup.momentum ?? this.momentum;
            const dampening = paramGroup.dampening ?? this.dampening;
            const weightDecay = paramGroup.weightDecay ?? this.weightDecay;
            const nesterov = paramGroup.nesterov ?? this.nesterov;
            for (const param of paramGroup.params) {
                if (!param.grad || !param.requiresGrad)
                    continue;
                let grad = param.grad.detach(), detachedParam = param.detach();
                // Apply weight decay (L2 regularization)
                if (weightDecay !== 0) {
                    grad = grad.add(detachedParam.mul(weightDecay));
                }
                // Apply momentum
                if (momentum !== 0) {
                    let buf = this.momentumBuffers.get(param);
                    if (!buf) {
                        // First time: initialize momentum buffer with current gradient
                        buf = grad.clone();
                        this.momentumBuffers.set(param, buf);
                    }
                    else {
                        // Update momentum buffer: buf = momentum * buf + (1 - dampening) * grad
                        buf = buf.mul(momentum).add(grad.mul(1 - dampening));
                        this.momentumBuffers.set(param, buf);
                    }
                    if (nesterov) {
                        // Nesterov momentum: grad = grad + momentum * buf
                        grad = grad.add(buf.mul(momentum));
                    }
                    else {
                        // Standard momentum: use momentum buffer as gradient
                        grad = buf;
                    }
                }
                // Update parameter: param = param - lr * grad
                const newParam = detachedParam.sub(grad.mul(lr));
                param.replace(newParam);
            }
        }
    }
}
exports.SGD = SGD;
class Adam extends BaseOptimizer {
    lr;
    betas;
    eps;
    weightDecay;
    momentumBuffers = new Map(); // First moment (m_t)
    velocityBuffers = new Map(); // Second moment (v_t)
    stepCounts = new Map();
    constructor(params, options) {
        super(params);
        this.lr = options?.lr ?? 0.001;
        this.betas = options?.betas ?? [0.9, 0.999];
        this.eps = options?.eps ?? 1e-8;
        this.weightDecay = options?.weightDecay ?? 0;
    }
    step() {
        for (const paramGroup of this.paramGroups) {
            const lr = paramGroup.lr ?? this.lr;
            const betas = paramGroup.betas ?? this.betas;
            const eps = paramGroup.eps ?? this.eps;
            const weightDecay = paramGroup.weightDecay ?? this.weightDecay;
            for (const param of paramGroup.params) {
                if (!param.grad || !param.requiresGrad)
                    continue;
                // Get current step for param, initialize if has not step before
                const stepCount = (this.stepCounts.get(param) ?? 0) + 1;
                this.stepCounts.set(param, stepCount);
                // Bias correction factors
                const [beta1, beta2] = betas;
                const biasCorrection1 = 1 - Math.pow(beta1, stepCount);
                const biasCorrection2 = 1 - Math.pow(beta2, stepCount);
                let grad = param.grad.detach(), detachedParam = param.detach();
                // Apply weight decay (L2 regularization)
                if (weightDecay !== 0) {
                    grad = grad.add(detachedParam.mul(weightDecay));
                }
                // Get or initialize first moment buffer (momentum)
                let momentumBuffer = this.momentumBuffers.get(param);
                if (!momentumBuffer) {
                    momentumBuffer = core_1.Tensor.zerosLike(grad); // Initialize with zeros (same shape as grad)
                    this.momentumBuffers.set(param, momentumBuffer);
                }
                // Get or initialize second moment buffer (velocity)
                let velocityBuffer = this.velocityBuffers.get(param);
                if (!velocityBuffer) {
                    velocityBuffer = core_1.Tensor.zerosLike(grad); // Initialize with zeros (same shape as grad)
                    this.velocityBuffers.set(param, velocityBuffer);
                }
                // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                momentumBuffer = momentumBuffer.mul(beta1).add(grad.mul(1 - beta1));
                this.momentumBuffers.set(param, momentumBuffer);
                // Update biased second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                velocityBuffer = velocityBuffer.mul(beta2).add(grad.pow(2).mul(1 - beta2));
                this.velocityBuffers.set(param, velocityBuffer);
                // Compute bias-corrected first moment: m_hat_t = m_t / (1 - beta1^t)
                const correctedMomentum = momentumBuffer.div(biasCorrection1);
                // Compute bias-corrected second moment: v_hat_t = v_t / (1 - beta2^t)
                const correctedVelocity = velocityBuffer.div(biasCorrection2);
                // Update parameters: theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)
                const denom = correctedVelocity.sqrt().add(eps);
                const stepSize = correctedMomentum.div(denom).mul(lr);
                const newParam = detachedParam.sub(stepSize);
                param.replace(newParam);
            }
        }
    }
}
exports.Adam = Adam;
class AdamW extends BaseOptimizer {
    lr;
    betas;
    eps;
    weightDecay;
    momentumBuffers = new Map(); // First moment (m_t)
    velocityBuffers = new Map(); // Second moment (v_t)
    stepCounts = new Map();
    constructor(params, options) {
        super(params);
        this.lr = options?.lr ?? 0.001;
        this.betas = options?.betas ?? [0.9, 0.999];
        this.eps = options?.eps ?? 1e-8;
        this.weightDecay = options?.weightDecay ?? 0.01;
    }
    step() {
        for (const paramGroup of this.paramGroups) {
            const lr = paramGroup.lr ?? this.lr;
            const betas = paramGroup.betas ?? this.betas;
            const eps = paramGroup.eps ?? this.eps;
            const weightDecay = paramGroup.weightDecay ?? this.weightDecay;
            for (const param of paramGroup.params) {
                if (!param.grad || !param.requiresGrad)
                    continue;
                // Get current step for param, initialize if has not step before
                const stepCount = (this.stepCounts.get(param) ?? 0) + 1;
                this.stepCounts.set(param, stepCount);
                // Bias correction factors
                const [beta1, beta2] = betas;
                const biasCorrection1 = 1 - Math.pow(beta1, stepCount);
                const biasCorrection2 = 1 - Math.pow(beta2, stepCount);
                let grad = param.grad.detach(), detachedParam = param.detach();
                // Apply weight decay (L2 regularization)
                detachedParam = detachedParam.sub(detachedParam.mul(weightDecay).mul(lr));
                // Get or initialize first moment buffer (momentum)
                let momentumBuffer = this.momentumBuffers.get(param);
                if (!momentumBuffer) {
                    momentumBuffer = core_1.Tensor.zerosLike(grad); // Initialize with zeros (same shape as grad)
                    this.momentumBuffers.set(param, momentumBuffer);
                }
                // Get or initialize second moment buffer (velocity)
                let velocityBuffer = this.velocityBuffers.get(param);
                if (!velocityBuffer) {
                    velocityBuffer = core_1.Tensor.zerosLike(grad); // Initialize with zeros (same shape as grad)
                    this.velocityBuffers.set(param, velocityBuffer);
                }
                // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                momentumBuffer = momentumBuffer.mul(beta1).add(grad.mul(1 - beta1));
                this.momentumBuffers.set(param, momentumBuffer);
                // Update biased second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                velocityBuffer = velocityBuffer.mul(beta2).add(grad.pow(2).mul(1 - beta2));
                this.velocityBuffers.set(param, velocityBuffer);
                // Compute bias-corrected first moment: m_hat_t = m_t / (1 - beta1^t)
                const correctedMomentum = momentumBuffer.div(biasCorrection1);
                // Compute bias-corrected second moment: v_hat_t = v_t / (1 - beta2^t)
                const correctedVelocity = velocityBuffer.div(biasCorrection2);
                // Update parameters: theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)
                const denom = correctedVelocity.sqrt().add(eps);
                const stepSize = correctedMomentum.div(denom).mul(lr);
                const newParam = detachedParam.sub(stepSize);
                param.replace(newParam);
            }
        }
    }
}
exports.AdamW = AdamW;
exports.Optim = {
    BaseOptimizer,
    SGD,
    Adam,
    AdamW
};

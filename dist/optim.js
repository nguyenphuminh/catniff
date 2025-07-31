"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Optim = void 0;
const core_1 = require("./core");
class SGD {
    params;
    momentumBuffers = new Map();
    lr;
    momentum;
    dampening;
    weightDecay;
    nesterov;
    constructor(params, options) {
        this.params = params;
        this.lr = options?.lr || 0.001;
        this.momentum = options?.momentum || 0;
        this.dampening = options?.dampening || 0;
        this.weightDecay = options?.weightDecay || 0;
        this.nesterov = options?.nesterov || false;
    }
    step() {
        for (const param of this.params) {
            if (!param.grad) {
                throw new Error("Can not apply SGD on empty grad");
            }
            let grad = param.grad.detach(), detachedParam = param.detach();
            // Apply weight decay (L2 regularization)
            if (this.weightDecay !== 0) {
                grad = grad.add(detachedParam.mul(this.weightDecay));
            }
            // Apply momentum
            if (this.momentum !== 0) {
                let buf = this.momentumBuffers.get(param);
                if (!buf) {
                    // First time: initialize momentum buffer with current gradient
                    buf = grad.clone();
                    this.momentumBuffers.set(param, buf);
                }
                else {
                    // Update momentum buffer: buf = momentum * buf + (1 - dampening) * grad
                    buf = buf.mul(this.momentum).add(grad.mul(1 - this.dampening));
                    this.momentumBuffers.set(param, buf);
                }
                if (this.nesterov) {
                    // Nesterov momentum: grad = grad + momentum * buf
                    grad = grad.add(buf.mul(this.momentum));
                }
                else {
                    // Standard momentum: use momentum buffer as gradient
                    grad = buf;
                }
            }
            // Update parameter: param = param - lr * grad
            const newParam = detachedParam.sub(grad.mul(this.lr));
            param.replace(newParam);
        }
    }
}
class Adam {
    params;
    momentumBuffers = new Map(); // First moment (m_t)
    velocityBuffers = new Map(); // Second moment (v_t)
    stepCount = 0;
    lr;
    betas;
    eps;
    weightDecay;
    constructor(params, options) {
        this.params = params;
        this.lr = options?.lr || 0.001;
        this.betas = options?.betas || [0.9, 0.999];
        this.eps = options?.eps || 1e-8;
        this.weightDecay = options?.weightDecay || 0;
    }
    step() {
        this.stepCount++;
        const beta1 = this.betas[0];
        const beta2 = this.betas[1];
        // Bias correction factors
        const biasCorrection1 = 1 - Math.pow(beta1, this.stepCount);
        const biasCorrection2 = 1 - Math.pow(beta2, this.stepCount);
        for (const param of this.params) {
            if (!param.grad) {
                throw new Error("Can not apply Adam on empty grad");
            }
            let grad = param.grad.detach(), detachedParam = param.detach();
            // Apply weight decay (L2 regularization)
            if (this.weightDecay !== 0) {
                grad = grad.add(detachedParam.mul(this.weightDecay));
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
            // Update biased first moment estimate: m_t = β1 * m_{t-1} + (1 - β1) * g_t
            momentumBuffer = momentumBuffer.mul(beta1).add(grad.mul(1 - beta1));
            this.momentumBuffers.set(param, momentumBuffer);
            // Update biased second moment estimate: v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
            velocityBuffer = velocityBuffer.mul(beta2).add(grad.pow(2).mul(1 - beta2));
            this.velocityBuffers.set(param, velocityBuffer);
            // Compute bias-corrected first moment: m̂_t = m_t / (1 - β1^t)
            const correctedMomentum = momentumBuffer.div(biasCorrection1);
            // Compute bias-corrected second moment: v̂_t = v_t / (1 - β2^t)
            const correctedVelocity = velocityBuffer.div(biasCorrection2);
            // Update parameters: θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
            const denom = correctedVelocity.sqrt().add(this.eps);
            const stepSize = correctedMomentum.div(denom).mul(this.lr);
            const newParam = detachedParam.sub(stepSize);
            param.replace(newParam);
        }
    }
}
class Optim {
    static SGD = SGD;
    static Adam = Adam;
}
exports.Optim = Optim;

"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Optim = void 0;
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
            if (typeof param.grad === "undefined") {
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
class Optim {
    static SGD = SGD;
}
exports.Optim = Optim;

"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LRScheduler = exports.StepLR = void 0;
class StepLR {
    optimizer;
    stepSize;
    gamma;
    lastEpoch;
    baseLR;
    constructor(optimizer, stepSize, gamma = 0.1, lastEpoch = -1) {
        this.optimizer = optimizer;
        this.stepSize = stepSize;
        this.gamma = gamma;
        this.lastEpoch = lastEpoch;
        this.baseLR = this.optimizer.lr;
    }
    step(epoch) {
        if (typeof epoch === "undefined") {
            this.lastEpoch++;
            epoch = this.lastEpoch;
        }
        else {
            this.lastEpoch = epoch;
        }
        this.optimizer.lr = this.baseLR * this.gamma ** Math.floor(epoch / this.stepSize);
    }
}
exports.StepLR = StepLR;
exports.LRScheduler = {
    StepLR
};

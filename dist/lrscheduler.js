"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LRScheduler = exports.StepLR = void 0;
class StepLR {
    optimizer;
    stepSize;
    gamma;
    lastEpoch;
    baseLR;
    baseGroupLRs;
    constructor(optimizer, stepSize, gamma = 0.1, lastEpoch = -1) {
        this.optimizer = optimizer;
        this.stepSize = stepSize;
        this.gamma = gamma;
        this.lastEpoch = lastEpoch;
        this.baseLR = optimizer.lr;
        this.baseGroupLRs = this.optimizer.paramGroups.map(paramGroup => paramGroup.lr ?? this.optimizer.lr);
    }
    step(epoch) {
        if (typeof epoch === "undefined") {
            this.lastEpoch++;
            epoch = this.lastEpoch;
        }
        else {
            this.lastEpoch = epoch;
        }
        // Update LR of each group
        for (let index = 0; index < this.baseGroupLRs.length; index++) {
            this.optimizer.paramGroups[index].lr = this.baseGroupLRs[index] * this.gamma ** Math.floor(epoch / this.stepSize);
        }
        // Update default LR
        this.optimizer.lr = this.baseLR * this.gamma ** Math.floor(epoch / this.stepSize);
    }
}
exports.StepLR = StepLR;
exports.LRScheduler = {
    StepLR
};

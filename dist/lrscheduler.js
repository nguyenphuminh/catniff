"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LRScheduler = exports.SequentialLR = exports.CosineAnnealingLR = exports.LinearLR = exports.StepLR = void 0;
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
    step() {
        this.lastEpoch++;
        // Update LR of each group
        for (let index = 0; index < this.baseGroupLRs.length; index++) {
            this.optimizer.paramGroups[index].lr = this.baseGroupLRs[index] * this.gamma ** Math.floor(this.lastEpoch / this.stepSize);
        }
        // Update default LR
        this.optimizer.lr = this.baseLR * this.gamma ** Math.floor(this.lastEpoch / this.stepSize);
    }
}
exports.StepLR = StepLR;
class LinearLR {
    optimizer;
    startFactor;
    endFactor;
    totalIters;
    lastEpoch;
    baseLR;
    baseGroupLRs;
    constructor(optimizer, startFactor = 0.3333333333333333, endFactor = 1, totalIters = 5, lastEpoch = -1) {
        this.optimizer = optimizer;
        this.startFactor = startFactor;
        this.endFactor = endFactor;
        this.totalIters = totalIters;
        this.lastEpoch = lastEpoch;
        this.baseLR = optimizer.lr;
        this.baseGroupLRs = this.optimizer.paramGroups.map(paramGroup => paramGroup.lr ?? this.optimizer.lr);
    }
    step() {
        this.lastEpoch++;
        // Clamp under total allowed iterations
        const t = Math.min(this.lastEpoch, this.totalIters);
        // Precalculate factor
        const factor = this.startFactor + (t / this.totalIters) * (this.endFactor - this.startFactor);
        // Update LR of each group
        for (let index = 0; index < this.baseGroupLRs.length; index++) {
            this.optimizer.paramGroups[index].lr = this.baseGroupLRs[index] * factor;
        }
        // Update default LR
        this.optimizer.lr = this.baseLR * factor;
    }
}
exports.LinearLR = LinearLR;
class CosineAnnealingLR {
    optimizer;
    TMax;
    etaMin;
    lastEpoch;
    baseLR;
    baseGroupLRs;
    constructor(optimizer, TMax, etaMin = 0, lastEpoch = -1) {
        this.optimizer = optimizer;
        this.TMax = TMax;
        this.etaMin = etaMin;
        this.lastEpoch = lastEpoch;
        this.baseLR = optimizer.lr;
        this.baseGroupLRs = this.optimizer.paramGroups.map(paramGroup => paramGroup.lr ?? this.optimizer.lr);
    }
    step() {
        this.lastEpoch++;
        const cosine = (1 + Math.cos((this.lastEpoch * Math.PI) / this.TMax)) / 2;
        for (let index = 0; index < this.baseGroupLRs.length; index++) {
            this.optimizer.paramGroups[index].lr = this.etaMin + (this.baseGroupLRs[index] - this.etaMin) * cosine;
        }
        this.optimizer.lr = this.etaMin + (this.baseLR - this.etaMin) * cosine;
    }
}
exports.CosineAnnealingLR = CosineAnnealingLR;
class SequentialLR {
    optimizer;
    schedulers;
    milestones;
    lastEpoch;
    constructor(optimizer, schedulers, milestones, lastEpoch = -1) {
        this.optimizer = optimizer;
        this.schedulers = schedulers;
        this.milestones = milestones;
        this.lastEpoch = lastEpoch;
    }
    step() {
        this.lastEpoch++;
        let schedulerIndex = this.schedulers.length - 1; // default to last
        for (let index = 0; index < this.milestones.length; index++) {
            if (this.lastEpoch < this.milestones[index]) {
                schedulerIndex = index;
                break;
            }
        }
        this.schedulers[schedulerIndex].step();
    }
}
exports.SequentialLR = SequentialLR;
exports.LRScheduler = {
    StepLR,
    LinearLR,
    CosineAnnealingLR,
    SequentialLR
};

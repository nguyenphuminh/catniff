import { OptimizerWithLR } from "./optim";

export class StepLR {
    public optimizer: OptimizerWithLR;
    public stepSize: number;
    public gamma: number;
    public lastEpoch: number;
    public baseLR: number;
    public baseGroupLRs: number[];

    constructor(optimizer: OptimizerWithLR, stepSize: number, gamma = 0.1, lastEpoch = -1) {
        this.optimizer = optimizer;
        this.stepSize = stepSize;
        this.gamma = gamma;
        this.lastEpoch = lastEpoch;
        this.baseLR = optimizer.lr;
        this.baseGroupLRs = this.optimizer.paramGroups.map(paramGroup => paramGroup.lr ?? this.optimizer.lr);
    }

    step(epoch?: number) {
        if (typeof epoch === "undefined") {
            this.lastEpoch++;
            epoch = this.lastEpoch;
        } else {
            this.lastEpoch = epoch;
        }

        // Update LR of each group
        for (let index = 0; index < this.baseGroupLRs.length; index++) {
            this.optimizer.paramGroups[index].lr = this.baseGroupLRs[index] * this.gamma**Math.floor(epoch / this.stepSize);
        }

        // Update default LR
        this.optimizer.lr = this.baseLR * this.gamma**Math.floor(epoch / this.stepSize);
    }
}

export const LRScheduler = {
    StepLR
}

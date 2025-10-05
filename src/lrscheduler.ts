import { BaseOptimizer } from "./optim";

export class StepLR {
    public optimizer: BaseOptimizer;
    public stepSize: number;
    public gamma: number;
    public lastEpoch: number;
    public baseLR: number;

    constructor(optimizer: BaseOptimizer, stepSize: number, gamma = 0.1, lastEpoch = -1) {
        this.optimizer = optimizer;
        this.stepSize = stepSize;
        this.gamma = gamma;
        this.lastEpoch = lastEpoch;
        this.baseLR = this.optimizer.lr;
    }

    step(epoch?: number) {
        if (typeof epoch === "undefined") {
            this.lastEpoch++;
            epoch = this.lastEpoch;
        } else {
            this.lastEpoch = epoch;
        }

        this.optimizer.lr = this.baseLR * this.gamma**Math.floor(epoch / this.stepSize);
    }
}

export const LRScheduler = {
    StepLR
}

import { Tensor } from "./core";
export interface Backend {
    create(tensor: Tensor): void;
    transfer(tensor: Tensor): Tensor;
}

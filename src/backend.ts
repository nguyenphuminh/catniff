import { Tensor } from "./core";

export interface Backend {
    transfer(tensor: Tensor): Tensor;
}

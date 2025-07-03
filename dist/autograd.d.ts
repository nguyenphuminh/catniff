import { Tensor } from "./tensor";
export declare enum OP {
    NONE = 0,
    ADD = 1,
    SUB = 2,
    MUL = 3,
    POW = 4,
    DIV = 5,
    NEG = 6,
    EXP = 7,
    LOG = 8,
    RELU = 9,
    SIGMOID = 10,
    TANH = 11
}
export declare class Node {
    value: Tensor;
    shape: number[];
    grad: Tensor;
    children: Node[];
    op: OP;
    feedBackward: Function;
    constructor(value: Tensor, children?: Node[], op?: OP);
    add(other: Node | number): Node;
    sub(other: Node | number): Node;
    mul(other: Node | number): Node;
    pow(other: Node | number): Node;
    div(other: Node | number): Node;
    neg(): Node;
    exp(): Node;
    log(): Node;
    relu(): Node;
    sigmoid(): Node;
    tanh(): Node;
    backward(): void;
    forceNode(value: Node | number): Node;
}

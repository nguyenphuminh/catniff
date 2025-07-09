import { Tensor } from "./tensor";
export declare enum OP {
    NONE = 0,
    ADD = 1,
    SUB = 2,
    MUL = 3,
    POW = 4,
    DIV = 5,
    GE = 6,
    LE = 7,
    GT = 8,
    LT = 9,
    EQ = 10,
    NEG = 11,
    ABS = 12,
    SIGN = 13,
    SIN = 14,
    COS = 15,
    TAN = 16,
    ASIN = 17,
    ACOS = 18,
    ATAN = 19,
    SINH = 20,
    COSH = 21,
    ASINH = 22,
    ACOSH = 23,
    ATANH = 24,
    SQRT = 25,
    EXP = 26,
    LOG = 27,
    LOG2 = 28,
    LOG10 = 29,
    LOG1P = 30,
    RELU = 31,
    SIGMOID = 32,
    TANH = 33,
    T = 34,
    MM = 35
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
    ge(other: Node | number): Node;
    le(other: Node | number): Node;
    gt(other: Node | number): Node;
    lt(other: Node | number): Node;
    eq(other: Node | number): Node;
    neg(): Node;
    abs(): Node;
    sign(): Node;
    sin(): Node;
    cos(): Node;
    tan(): Node;
    asin(): Node;
    acos(): Node;
    atan(): Node;
    sinh(): Node;
    cosh(): Node;
    asinh(): Node;
    acosh(): Node;
    atanh(): Node;
    sqrt(): Node;
    exp(): Node;
    log(): Node;
    log2(): Node;
    log10(): Node;
    log1p(): Node;
    relu(): Node;
    sigmoid(): Node;
    tanh(): Node;
    t(): Node;
    mm(other: Node | number): Node;
    backward(): void;
    static forceNode(value: Node | number): Node;
    static addGrad(node: Node, accumGrad: Tensor): void;
}

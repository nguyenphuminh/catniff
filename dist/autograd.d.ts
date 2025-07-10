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
    LOGICALAND = 11,
    LOGICALOR = 12,
    LOGICALXOR = 13,
    LOGICALNOT = 14,
    BITWISEAND = 15,
    BITWISEOR = 16,
    BITWISEXOR = 17,
    BITWISENOT = 18,
    BITWISELEFTSHIFT = 19,
    BITWISERIGHTSHIFT = 20,
    NEG = 21,
    ABS = 22,
    SIGN = 23,
    SIN = 24,
    COS = 25,
    TAN = 26,
    ASIN = 27,
    ACOS = 28,
    ATAN = 29,
    SINH = 30,
    COSH = 31,
    ASINH = 32,
    ACOSH = 33,
    ATANH = 34,
    SQRT = 35,
    EXP = 36,
    LOG = 37,
    LOG2 = 38,
    LOG10 = 39,
    LOG1P = 40,
    RELU = 41,
    SIGMOID = 42,
    TANH = 43,
    T = 44,
    MM = 45,
    DOT = 46
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
    logicalAnd(other: Node | number): Node;
    logicalOr(other: Node | number): Node;
    logicalXor(other: Node | number): Node;
    logicalNot(): Node;
    bitwiseAnd(other: Node | number): Node;
    bitwiseOr(other: Node | number): Node;
    bitwiseXor(other: Node | number): Node;
    bitwiseNot(): Node;
    bitwiseLeftShift(other: Node | number): Node;
    bitwiseRightShift(other: Node | number): Node;
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
    dot(other: Node | number): Node;
    backward(): void;
    static forceNode(value: Node | number): Node;
    static addGrad(node: Node, accumGrad: Tensor): void;
}

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
    DOT = 45,
    MM = 46,
    MV = 47,
    MATMUL = 48
}
export declare class Node {
    value: Tensor;
    shape: number[];
    grad: Tensor;
    children: Node[];
    op: OP;
    feedBackward: Function;
    constructor(value: Tensor, children?: Node[], op?: OP);
    add(other: Node | Tensor): Node;
    sub(other: Node | Tensor): Node;
    mul(other: Node | Tensor): Node;
    pow(other: Node | Tensor): Node;
    div(other: Node | Tensor): Node;
    ge(other: Node | Tensor): Node;
    le(other: Node | Tensor): Node;
    gt(other: Node | Tensor): Node;
    lt(other: Node | Tensor): Node;
    eq(other: Node | Tensor): Node;
    logicalAnd(other: Node | Tensor): Node;
    logicalOr(other: Node | Tensor): Node;
    logicalXor(other: Node | Tensor): Node;
    logicalNot(): Node;
    bitwiseAnd(other: Node | Tensor): Node;
    bitwiseOr(other: Node | Tensor): Node;
    bitwiseXor(other: Node | Tensor): Node;
    bitwiseNot(): Node;
    bitwiseLeftShift(other: Node | Tensor): Node;
    bitwiseRightShift(other: Node | Tensor): Node;
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
    dot(other: Node | Tensor): Node;
    mm(other: Node | Tensor): Node;
    mv(other: Node | Tensor): Node;
    matmul(other: Node | Tensor): Node;
    backward(): void;
    static forceNode(value: Node | Tensor): Node;
    static addGrad(node: Node, accumGrad: Tensor): void;
}

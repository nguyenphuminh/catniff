export declare enum OP {
    NONE = 0,
    ADD = 1,
    MUL = 2,
    POW = 3,
    DIV = 4,
    NEG = 5,
    EXP = 6,
    LOG = 7,
    RELU = 8,
    SIGMOID = 9,
    TANH = 10
}
export declare class Node {
    value: number;
    grad: number;
    children: Node[];
    feedBackward: Function;
    op: OP;
    constructor(value: number, children?: Node[], op?: OP);
    add(other: Node | number): Node;
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

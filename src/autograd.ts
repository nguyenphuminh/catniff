import { Tensor, TensorMath } from "./tensor";

const {
    add,
    sub,
    mul,
    pow,
    div,
    neg,
    exp,
    log,
    relu,
    sigmoid,
    tanh,
    ge,
    t,
    mm
} = TensorMath;

export enum OP {
    NONE,
    ADD,
    SUB,
    MUL,
    POW,
    DIV,
    NEG,
    EXP,
    LOG,
    RELU,
    SIGMOID,
    TANH,
    T,
    MM
}

export class Node {
    public value: Tensor;
    public shape: number[];
    public grad: Tensor;
    public children: Node[];
    public op: OP;
    public feedBackward: Function;

    constructor(value: Tensor, children: Node[] = [], op: OP = OP.NONE) {
        this.value = value;
        this.shape = TensorMath.getShape(value);
        this.grad = TensorMath.create(0, this.shape);
        this.children = children;
        this.op = op;
        this.feedBackward = () => {};
    }

    add(other: Node | number): Node {
        other = Node.forceNode(other);
        const out = new Node(add(this.value, other.value), [this, other], OP.ADD);

        out.feedBackward = () => {
            // x + y d/dx = 1, note that we apply the chain rule continuously so out.grad is multiplied into our derivative
            Node.addGrad(this, out.grad);
            // x + y d/dy = 1
            Node.addGrad(other, out.grad);
        }

        return out;
    }

    sub(other: Node | number): Node {
        other = Node.forceNode(other);
        const out = new Node(sub(this.value, other.value), [this, other], OP.SUB);

        out.feedBackward = () => {
            // x - y d/dx = 1
            Node.addGrad(this, out.grad);
            // x - y d/dy = -1
            Node.addGrad(other, neg(out.grad));
        }

        return out;
    }

    mul(other: Node | number): Node {
        other = Node.forceNode(other);
        const out = new Node(mul(this.value, other.value), [this, other], OP.MUL);

        out.feedBackward = () => {
            // x * y d/dx = y
            Node.addGrad(this, mul(out.grad, other.value));
            // x * y d/dy = x
            Node.addGrad(other, mul(out.grad, this.value));
        }

        return out;
    }

    pow(other: Node | number): Node {
        if (other instanceof Node) {
            const out = new Node(pow(this.value, other.value), [this, other], OP.POW);

            out.feedBackward = () => {
                // x^a d/dx = a*x^(a-1)
                Node.addGrad(this, mul(out.grad, mul(other.value, pow(this.value, sub(other.value, 1)))));
                // x^a d/da = x^a*lnx
                Node.addGrad(other, mul(out.grad, mul(pow(this.value, other.value), log(this.value))));
            }

            return out;
        }

        const out = new Node(pow(this.value, other), [this], OP.POW);

        out.feedBackward = () => {
            Node.addGrad(this, mul(out.grad, mul(other, pow(this.value, sub(other, 1)))));
        }

        return out;
    }

    div(other: Node | number): Node {
        other = Node.forceNode(other);
        const out = new Node(div(this.value, other.value), [this, other], OP.DIV);

        out.feedBackward = () => {
            // x/y d/dx = 1/y
            Node.addGrad(this, div(out.grad, other.value));
            // x/y d/dy = -x/y^2
            Node.addGrad(other, mul(out.grad, div(neg(this.value), pow(other.value, 2))));
        }

        return out;
    }

    neg(): Node {
        const out = new Node(neg(this.value), [this], OP.NEG);

        out.feedBackward = () => {
            // -x d/dx = -1
            Node.addGrad(this, neg(out.grad));
        }

        return out;
    }

    exp(): Node {
        const expResult = exp(this.value);
        const out = new Node(expResult, [this], OP.EXP);

        out.feedBackward = () => {
            // e^x d/dx = e^x
            Node.addGrad(this, mul(out.grad, expResult));
        }

        return out;
    }

    log(): Node {
        const out = new Node(log(this.value), [this], OP.LOG);

        out.feedBackward = () => {
            // lnx d/dx = 1/x
            Node.addGrad(this, div(out.grad, this.value));
        }

        return out;
    }

    relu(): Node {
        const out = new Node(relu(this.value), [this], OP.RELU);

        out.feedBackward = () => {
            Node.addGrad(this, mul(out.grad, ge(this.value, 0)));
        };

        return out;
    }

    sigmoid(): Node {
        const sigmoidResult = sigmoid(this.value);
        const out = new Node(sigmoidResult, [this], OP.SIGMOID);

        out.feedBackward = () => {
            Node.addGrad(this, mul(mul(out.grad, sigmoidResult), sub(1, sigmoidResult)));
        };

        return out;
    }

    tanh(): Node {
        const tanhResult = tanh(this.value);
        const out = new Node(tanhResult, [this], OP.TANH);

        out.feedBackward = () => {
            Node.addGrad(this, mul(out.grad, sub(1, mul(tanhResult, tanhResult))));
        };

        return out;
    }

    t(): Node {
        const out = new Node(t(this.value), [this], OP.T);

        out.feedBackward = () => {
            Node.addGrad(this, t(out.grad));
        };

        return out;
    }

    mm(other: Node | number): Node {
        other = Node.forceNode(other);
        const out = new Node(mm(this.value, other.value), [this, other], OP.MM);

        out.feedBackward = () => {
            Node.addGrad(this, mm(out.grad, t(other.value)));
            Node.addGrad(other, mm(t(this.value), out.grad));
        }

        return out;
    }

    backward() {
        // Build topological order
        const topo: Node[] = [];
        const visited: Set<Node> = new Set();

        function build(node: Node) {
            if (!visited.has(node)) {
                visited.add(node);
                node.grad = TensorMath.create(0, node.shape);

                for (let child of node.children) build(child);
                topo.push(node);
            }
        }

        build(this);

        // Feed backward to calculate gradient
        this.grad = TensorMath.create(1, this.shape); // Derivative of itself with respect to itself

        for (let index = topo.length - 1; index > -1; index--) {
            topo[index].feedBackward();
        }
    }

    static forceNode(value: Node | number): Node {
        if (value instanceof Node) return value;

        return new Node(value);
    }

    static addGrad(node: Node, accumGrad: Tensor) {
        const axesToSqueeze = [];
        const axesToReduce = [];

        const shape = node.shape;
        const gradShape = TensorMath.getShape(accumGrad);

        const paddedDims = gradShape.length - shape.length;

        for (let i = 0; i < paddedDims; i++) {
            axesToReduce.push(i);
            axesToSqueeze.push(i);
        }

        for (let i = 0; i < shape.length; i++) {
            if (shape[i] === 1 && gradShape[i+paddedDims] > 1) {
                axesToReduce.push(i+paddedDims);
            }
        }

        const reducedGrad = TensorMath.sum(accumGrad, axesToReduce, true);
        const squeezedGrad = TensorMath.squeeze(reducedGrad, axesToSqueeze);

        node.grad = add(squeezedGrad, node.grad);
    }
}

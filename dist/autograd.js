"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Node = exports.OP = void 0;
const tensor_1 = require("./tensor");
const { add, sub, mul, pow, div, neg, exp, log, relu, sigmoid, tanh, ge } = tensor_1.TensorMath;
var OP;
(function (OP) {
    OP[OP["NONE"] = 0] = "NONE";
    OP[OP["ADD"] = 1] = "ADD";
    OP[OP["SUB"] = 2] = "SUB";
    OP[OP["MUL"] = 3] = "MUL";
    OP[OP["POW"] = 4] = "POW";
    OP[OP["DIV"] = 5] = "DIV";
    OP[OP["NEG"] = 6] = "NEG";
    OP[OP["EXP"] = 7] = "EXP";
    OP[OP["LOG"] = 8] = "LOG";
    OP[OP["RELU"] = 9] = "RELU";
    OP[OP["SIGMOID"] = 10] = "SIGMOID";
    OP[OP["TANH"] = 11] = "TANH";
})(OP || (exports.OP = OP = {}));
class Node {
    value;
    shape;
    grad;
    children;
    op;
    feedBackward;
    constructor(value, children = [], op = OP.NONE) {
        this.value = value;
        this.shape = tensor_1.TensorMath.getShape(value);
        this.grad = tensor_1.TensorMath.create(0, this.shape);
        this.children = children;
        this.op = op;
        this.feedBackward = () => { };
    }
    add(other) {
        other = Node.forceNode(other);
        const out = new Node(add(this.value, other.value), [this, other], OP.ADD);
        out.feedBackward = () => {
            // x + y d/dx = 1, note that we apply the chain rule continuously so out.grad is multiplied into our derivative
            Node.addGrad(this, out.grad);
            // x + y d/dy = 1
            Node.addGrad(other, out.grad);
        };
        return out;
    }
    sub(other) {
        other = Node.forceNode(other);
        const out = new Node(sub(this.value, other.value), [this, other], OP.SUB);
        out.feedBackward = () => {
            // x - y d/dx = 1
            Node.addGrad(this, out.grad);
            // x - y d/dy = -1
            Node.addGrad(other, neg(out.grad));
        };
        return out;
    }
    mul(other) {
        other = Node.forceNode(other);
        const out = new Node(mul(this.value, other.value), [this, other], OP.MUL);
        out.feedBackward = () => {
            // x * y d/dx = y
            Node.addGrad(this, mul(out.grad, other.value));
            // x + y d/dy = x
            Node.addGrad(other, mul(out.grad, this.value));
        };
        return out;
    }
    pow(other) {
        if (other instanceof Node) {
            const out = new Node(pow(this.value, other.value), [this, other], OP.POW);
            out.feedBackward = () => {
                // x^a d/dx = a*x^(a-1)
                Node.addGrad(this, mul(out.grad, mul(other.value, pow(this.value, sub(other.value, 1)))));
                // x^a d/da = x^a*lnx
                Node.addGrad(other, mul(out.grad, mul(pow(this.value, other.value), log(this.value))));
            };
            return out;
        }
        const out = new Node(pow(this.value, other), [this], OP.POW);
        out.feedBackward = () => {
            Node.addGrad(this, mul(out.grad, mul(other, pow(this.value, sub(other, 1)))));
        };
        return out;
    }
    div(other) {
        other = Node.forceNode(other);
        const out = new Node(div(this.value, other.value), [this, other], OP.DIV);
        out.feedBackward = () => {
            // x/y d/dx = 1/y
            Node.addGrad(this, div(out.grad, other.value));
            // x/y d/dy = -x/y^2
            Node.addGrad(other, mul(out.grad, div(neg(this.value), pow(other.value, 2))));
        };
        return out;
    }
    neg() {
        const out = new Node(neg(this.value), [this], OP.NEG);
        out.feedBackward = () => {
            // -x d/dx = -1
            Node.addGrad(this, neg(out.grad));
        };
        return out;
    }
    exp() {
        const expResult = exp(this.value);
        const out = new Node(expResult, [this], OP.EXP);
        out.feedBackward = () => {
            // e^x d/dx = e^x
            Node.addGrad(this, mul(out.grad, expResult));
        };
        return out;
    }
    log() {
        const out = new Node(log(this.value), [this], OP.LOG);
        out.feedBackward = () => {
            // lnx d/dx = 1/x
            Node.addGrad(this, div(out.grad, this.value));
        };
        return out;
    }
    relu() {
        const out = new Node(relu(this.value), [this], OP.RELU);
        out.feedBackward = () => {
            Node.addGrad(this, mul(out.grad, ge(this.value, 0)));
        };
        return out;
    }
    sigmoid() {
        const sigmoidResult = sigmoid(this.value);
        const out = new Node(sigmoidResult, [this], OP.SIGMOID);
        out.feedBackward = () => {
            Node.addGrad(this, mul(mul(out.grad, sigmoidResult), sub(1, sigmoidResult)));
        };
        return out;
    }
    tanh() {
        const tanhResult = tanh(this.value);
        const out = new Node(tanhResult, [this], OP.TANH);
        out.feedBackward = () => {
            Node.addGrad(this, mul(out.grad, sub(1, mul(tanhResult, tanhResult))));
        };
        return out;
    }
    backward() {
        // Build topological order
        const topo = [];
        const visited = new Set();
        function build(node) {
            if (!visited.has(node)) {
                visited.add(node);
                node.grad = tensor_1.TensorMath.create(0, node.shape);
                for (let child of node.children)
                    build(child);
                topo.push(node);
            }
        }
        build(this);
        // Feed backward to calculate gradient
        this.grad = tensor_1.TensorMath.create(1, this.shape); // Derivative of itself with respect to itself
        for (let index = topo.length - 1; index > -1; index--) {
            topo[index].feedBackward();
        }
    }
    static forceNode(value) {
        if (value instanceof Node)
            return value;
        return new Node(value);
    }
    static addGrad(node, accumGrad) {
        const axesToSqueeze = [];
        const axesToReduce = [];
        const shape = node.shape;
        const gradShape = tensor_1.TensorMath.getShape(accumGrad);
        const paddedDims = gradShape.length - shape.length;
        for (let i = 0; i < paddedDims; i++) {
            axesToReduce.push(i);
            axesToSqueeze.push(i);
        }
        for (let i = 0; i < shape.length; i++) {
            if (shape[i] === 1 && gradShape[i + paddedDims] > 1) {
                axesToReduce.push(i + paddedDims);
            }
        }
        const reducedGrad = tensor_1.TensorMath.sum(accumGrad, axesToReduce, true);
        const squeezedGrad = tensor_1.TensorMath.squeeze(reducedGrad, axesToSqueeze);
        node.grad = add(squeezedGrad, node.grad);
    }
}
exports.Node = Node;

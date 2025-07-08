"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TensorMath = void 0;
class TensorMath {
    static create(num, shape) {
        if (shape.length === 0) {
            return num;
        }
        const [dim, ...rest] = shape;
        const out = [];
        for (let i = 0; i < dim; i++) {
            out.push(TensorMath.create(num, rest));
        }
        return out;
    }
    static getShape(tA) {
        const shape = [];
        let subA = tA;
        while (Array.isArray(subA)) {
            shape.push(subA.length);
            subA = subA[0];
        }
        return shape;
    }
    static padShape(tA, tB) {
        let dimA = TensorMath.getShape(tA).length;
        let dimB = TensorMath.getShape(tB).length;
        while (dimA < dimB) {
            dimA++;
            tA = [tA];
        }
        while (dimA > dimB) {
            dimB++;
            tB = [tB];
        }
        return [tA, tB];
    }
    static add(tA, tB) {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA + tB;
        }
        [tA, tB] = TensorMath.padShape(tA, tB);
        const outLen = Math.max(tA.length, tB.length);
        if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
            throw new Error("Inputs are incompatible tensors");
        }
        const result = [];
        for (let i = 0; i < outLen; i++) {
            const subA = tA[tA.length === 1 ? 0 : i];
            const subB = tB[tB.length === 1 ? 0 : i];
            result.push(TensorMath.add(subA, subB));
        }
        return result;
    }
    static sub(tA, tB) {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA - tB;
        }
        [tA, tB] = TensorMath.padShape(tA, tB);
        const outLen = Math.max(tA.length, tB.length);
        if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
            throw new Error("Inputs are incompatible tensors");
        }
        const result = [];
        for (let i = 0; i < outLen; i++) {
            const subA = tA[tA.length === 1 ? 0 : i];
            const subB = tB[tB.length === 1 ? 0 : i];
            result.push(TensorMath.sub(subA, subB));
        }
        return result;
    }
    static mul(tA, tB) {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA * tB;
        }
        [tA, tB] = TensorMath.padShape(tA, tB);
        const outLen = Math.max(tA.length, tB.length);
        if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
            throw new Error("Inputs are incompatible tensors");
        }
        const result = [];
        for (let i = 0; i < outLen; i++) {
            const subA = tA[tA.length === 1 ? 0 : i];
            const subB = tB[tB.length === 1 ? 0 : i];
            result.push(TensorMath.mul(subA, subB));
        }
        return result;
    }
    static pow(tA, tB) {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA ** tB;
        }
        [tA, tB] = TensorMath.padShape(tA, tB);
        const outLen = Math.max(tA.length, tB.length);
        if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
            throw new Error("Inputs are incompatible tensors");
        }
        const result = [];
        for (let i = 0; i < outLen; i++) {
            const subA = tA[tA.length === 1 ? 0 : i];
            const subB = tB[tB.length === 1 ? 0 : i];
            result.push(TensorMath.pow(subA, subB));
        }
        return result;
    }
    static div(tA, tB) {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA / tB;
        }
        [tA, tB] = TensorMath.padShape(tA, tB);
        const outLen = Math.max(tA.length, tB.length);
        if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
            throw new Error("Inputs are incompatible tensors");
        }
        const result = [];
        for (let i = 0; i < outLen; i++) {
            const subA = tA[tA.length === 1 ? 0 : i];
            const subB = tB[tB.length === 1 ? 0 : i];
            result.push(TensorMath.div(subA, subB));
        }
        return result;
    }
    static gt(tA, tB) {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA > tB ? 1 : 0;
        }
        [tA, tB] = TensorMath.padShape(tA, tB);
        const outLen = Math.max(tA.length, tB.length);
        if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
            throw new Error("Inputs are incompatible tensors");
        }
        const result = [];
        for (let i = 0; i < outLen; i++) {
            const subA = tA[tA.length === 1 ? 0 : i];
            const subB = tB[tB.length === 1 ? 0 : i];
            result.push(TensorMath.gt(subA, subB));
        }
        return result;
    }
    static lt(tA, tB) {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA < tB ? 1 : 0;
        }
        [tA, tB] = TensorMath.padShape(tA, tB);
        const outLen = Math.max(tA.length, tB.length);
        if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
            throw new Error("Inputs are incompatible tensors");
        }
        const result = [];
        for (let i = 0; i < outLen; i++) {
            const subA = tA[tA.length === 1 ? 0 : i];
            const subB = tB[tB.length === 1 ? 0 : i];
            result.push(TensorMath.lt(subA, subB));
        }
        return result;
    }
    static ge(tA, tB) {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA >= tB ? 1 : 0;
        }
        [tA, tB] = TensorMath.padShape(tA, tB);
        const outLen = Math.max(tA.length, tB.length);
        if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
            throw new Error("Inputs are incompatible tensors");
        }
        const result = [];
        for (let i = 0; i < outLen; i++) {
            const subA = tA[tA.length === 1 ? 0 : i];
            const subB = tB[tB.length === 1 ? 0 : i];
            result.push(TensorMath.ge(subA, subB));
        }
        return result;
    }
    static le(tA, tB) {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA <= tB ? 1 : 0;
        }
        [tA, tB] = TensorMath.padShape(tA, tB);
        const outLen = Math.max(tA.length, tB.length);
        if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
            throw new Error("Inputs are incompatible tensors");
        }
        const result = [];
        for (let i = 0; i < outLen; i++) {
            const subA = tA[tA.length === 1 ? 0 : i];
            const subB = tB[tB.length === 1 ? 0 : i];
            result.push(TensorMath.le(subA, subB));
        }
        return result;
    }
    static eq(tA, tB) {
        if (typeof tA === "number" && typeof tB === "number") {
            return tA === tB ? 1 : 0;
        }
        [tA, tB] = TensorMath.padShape(tA, tB);
        const outLen = Math.max(tA.length, tB.length);
        if (tA.length !== tB.length && tA.length !== 1 && tB.length !== 1) {
            throw new Error("Inputs are incompatible tensors");
        }
        const result = [];
        for (let i = 0; i < outLen; i++) {
            const subA = tA[tA.length === 1 ? 0 : i];
            const subB = tB[tB.length === 1 ? 0 : i];
            result.push(TensorMath.eq(subA, subB));
        }
        return result;
    }
    static neg(tA) {
        if (typeof tA === "number") {
            return -tA;
        }
        else {
            return tA.map(subA => TensorMath.neg(subA));
        }
    }
    static exp(tA) {
        if (typeof tA === "number") {
            return Math.exp(tA);
        }
        else {
            return tA.map(subA => TensorMath.exp(subA));
        }
    }
    static log(tA) {
        if (typeof tA === "number") {
            return Math.log(tA);
        }
        else {
            return tA.map(subA => TensorMath.log(subA));
        }
    }
    static relu(tA) {
        if (typeof tA === "number") {
            return Math.max(tA, 0);
        }
        else {
            return tA.map(subA => TensorMath.relu(subA));
        }
    }
    static sigmoid(tA) {
        if (typeof tA === "number") {
            return 1 / (1 + Math.exp(-tA));
        }
        else {
            return tA.map(subA => TensorMath.sigmoid(subA));
        }
    }
    static tanh(tA) {
        if (typeof tA === "number") {
            return Math.tanh(tA);
        }
        else {
            return tA.map(subA => TensorMath.tanh(subA));
        }
    }
    static squeezeAxis(tA, axis) {
        if (typeof tA === "number")
            return tA;
        if (axis === 0) {
            return tA[0];
        }
        else {
            return tA.map(slice => TensorMath.squeezeAxis(slice, axis - 1));
        }
    }
    static squeeze(tA, dims) {
        if (typeof tA === "number")
            return tA;
        if (typeof dims === "number") {
            dims = [dims];
        }
        if (typeof dims === "undefined") {
            const shape = TensorMath.getShape(tA);
            dims = [];
            for (let index = 0; index < shape.length; index++) {
                if (shape[index] === 1) {
                    dims.push(index);
                }
            }
        }
        dims = [...dims].sort((a, b) => b - a);
        let out = tA;
        for (const axis of dims) {
            out = TensorMath.squeezeAxis(out, axis);
        }
        return out;
    }
    static sumAxis(tA, axis) {
        if (typeof tA === "number")
            return tA;
        if (axis === 0) {
            let result = tA[0];
            for (let i = 1; i < tA.length; i++) {
                result = TensorMath.add(result, tA[i]);
            }
            return [result];
        }
        else {
            return tA.map(slice => TensorMath.sumAxis(slice, axis - 1));
        }
    }
    static sum(tA, dims, keepDims = false) {
        if (typeof tA === "number")
            return tA;
        if (typeof dims === "number") {
            dims = [dims];
        }
        if (typeof dims === "undefined") {
            dims = Array.from({ length: TensorMath.getShape(tA).length }, (_, index) => index);
        }
        dims = [...dims].sort((a, b) => b - a);
        let out = tA;
        for (const axis of dims) {
            out = TensorMath.sumAxis(out, axis);
        }
        return keepDims ? out : TensorMath.squeeze(out, dims);
    }
}
exports.TensorMath = TensorMath;

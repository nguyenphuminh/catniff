# Getting started

## Setup

Install through npm:
```
npm install catniff
```

## Tensors

Tensors in Catniff can be created by passing in a number or an nD array, and there are built-in methods that can be used to perform tensor arithmetic:
```js
const { Tensor } = require("catniff");

// Tensor init
const A = new Tensor([ 1, 2, 3 ]);
const B = new Tensor(3);

// Tensor addition (.val() returns the raw value rather than the tensor object)
console.log(A.add(B).val());
```

## Autograd

To compute the gradient wrt multiple variables of our mathematical expression, we can simply set `requiresGrad` to `true`:
```js
const { Tensor } = require("catniff");

const X = new Tensor(
    [
        [ 0.5, -1.0 ],
        [ 2.0,  0.0 ]
    ],
    { requiresGrad: true }
);

const Y = new Tensor(
    [
        [ 1.0, -2.0 ],
        [ 0.5,  1.5 ]
    ],
    { requiresGrad: true }
);

const D = X.sub(Y);
const E = D.exp();
const F = E.add(1);
const G = F.log();

G.backward();

// X.grad and Y.grad are tensor objects themselves, so we call .val() here to see their raw values
console.log(X.grad.val(), Y.grad.val());
```


# All APIs

Below is the specification for Catniff APIs. Note that undocumented APIs in the codebase are either unsafe or not ready for use other than internal.

## TensorValue

* Type `TensorValue` is either `number` or `TensorValue[]`, which means it represents either numbers or n-D number arrays.

## TensorOptions

* `TensorOptions` is an interface that contains options/configurations of a tensor passed into the `Tensor` class constructor (more on that later). It includes:
    * `shape?: readonly number[]`
    * `strides?: readonly number[]`
    * `grad?: Tensor`
    * `requiresGrad?: boolean`
    * `gradFn?: Function`
    * `children?: Tensor[]`

## Tensor

### Constructor

```ts
constructor(value: TensorValue, options: TensorOptions = {})
```

### Properties

* `public value: number[] | number`: Holds the tensor value as either a flat number array or a number, it is `Tensor.flatten(value)` behind the scenes.
* `public readonly shape: readonly number[]`: Holds the tensor shape, uses `options.shape` if provided, `Tensor.getShape(value)` otherwise.
* `public readonly strides: readonly number[]`: Holds the tensor strides, uses `options.strides` if provided, `Tensor.getStrides(this.shape)` otherwise.
* `public grad?: Tensor`: Holds the tensor gradient, uses `options.grad` if provided, `undefined` otherwise to save memory.
* `public requiresGrad: boolean`: Choose whether to do gradient-related operations behind the scenes, uses `options.requiresGrad` if provided, `false` otherwise.
* `public gradFn: Function`: Called when computing gradient all over the DAG, used to feed gradient to its child tensors, uses `options.gradFn` if provided, `() => {}` otherwise.
* `public children: Tensor[]`: Holds its child tensors, will be used when computing gradient, uses `options.children` if provided, `[]` otherwise.

Note: A good rule of thumb when using Catniff is to not mutate values passed into functions/methods. For example, this might introduce some unexpected behaviors:
```ts
const tensorVal = [1,2,3];
const tensor = new Tensor(tensorVal);
tensorVal[0] = 4; // This would change tensorVal.value too
```

because Catniff try not to allocate new arrays and use the argument if possible to save memory and performance.

### Methods

All autograd-supported tensor arithmetic methods:

* `add(other: TensorValue | Tensor): Tensor`: Returns `this` added with `other` element-wise. If `other` is a `TensorValue`, it will be converted to a `Tensor` with `Tensor.forceTensor(other)`, and this rule will apply for other element-wise ops as well. Two tensors of different shapes and sizes will get broadcasted if they are compatible, or else the function will throw an error.
* `sub(other: TensorValue | Tensor): Tensor`: Returns `this` subtracted by `other` element-wise.
* `mul(other: TensorValue | Tensor): Tensor`: Returns `this` multiplied with `other` element-wise.
* `pow(other: TensorValue | Tensor): Tensor`: Returns `this` raised to the power of `other` element-wise.
* `div(other: TensorValue | Tensor): Tensor`: Returns `this` divided by `other` element-wise.
* `ge(other: TensorValue | Tensor): Tensor`: Returns `this` greater than or equal to `other` element-wise (1 if true, 0 if false).
* `le(other: TensorValue | Tensor): Tensor`: Returns `this` less than or equal to `other` element-wise (1 if true, 0 if false).
* `gt(other: TensorValue | Tensor): Tensor`: Returns `this` greater than `other` element-wise (1 if true, 0 if false).
* `lt(other: TensorValue | Tensor): Tensor`: Returns `this` less than `other` element-wise (1 if true, 0 if false).
* `eq(other: TensorValue | Tensor): Tensor`: Returns `this` equal to `other` element-wise (1 if true, 0 if false).
* `logicalAnd(other: TensorValue | Tensor): Tensor`: Returns `this` logical and `other` element-wise (1 if both are 1, 0 otherwise).
* `logicalOr(other: TensorValue | Tensor): Tensor`: Returns `this` logical or `other` element-wise (1 if either are 1, 0 otherwise).
* `logicalXor(other: TensorValue | Tensor): Tensor`: Returns `this` logical xor `other` element-wise (1 if both are not the same bit, 0 otherwise).
* `logicalNot(): Tensor`: Returns logical not of `this` element-wise (1 if 0, 0 if 1).
* `bitwiseAnd(other: TensorValue | Tensor): Tensor`: Returns `this` bitwise and `other` element-wise.
* `bitwiseOr(other: TensorValue | Tensor): Tensor`: Returns `this` bitwise or `other` element-wise.
* `bitwiseXor(other: TensorValue | Tensor): Tensor`: Returns `this` bitwise xor `other` element-wise.
* `bitwiseNot(): Tensor`: Returns bitwise not of `this` element-wise.
* `bitwiseLeftShift(other: TensorValue | Tensor): Tensor`: Returns `this` bitwise left shift `other` element-wise.
* `bitwiseRightShift(other: TensorValue | Tensor): Tensor`: Returns `this` bitwise right shift `other` element-wise.
* `neg(): Tensor`: Returns negative of `this` element-wise.
* `abs(): Tensor`: Returns absolute of `this` element-wise.
* `sign(): Tensor`: Returns sign of `this` element-wise.
* `sin(): Tensor`: Returns sin of `this` element-wise.
* `cos(): Tensor`: Returns cos of `this` element-wise.
* `tan(): Tensor`: Returns tan of `this` element-wise.
* `asin(): Tensor`: Returns asin of `this` element-wise.
* `acos(): Tensor`: Returns acos of `this` element-wise.
* `atan(): Tensor`: Returns atan of `this` element-wise.
* `sinh(): Tensor`: Returns sinh of `this` element-wise.
* `cosh(): Tensor`: Returns cosh of `this` element-wise.
* `asinh(): Tensor`: Returns asinh of `this` element-wise.
* `acosh(): Tensor`: Returns acosh of `this` element-wise.
* `atanh(): Tensor`: Returns atanh of `this` element-wise.
* `sqrt(): Tensor`: Returns square root of `this` element-wise.
* `exp(): Tensor`: Returns e raised to the power of `this` element-wise.
* `log(): Tensor`: Returns natural log of `this` element-wise.
* `log2(): Tensor`: Returns log base 2 of `this` element-wise.
* `log10(): Tensor`: Returns log base 10 of `this` element-wise.
* `log1p(): Tensor`: Returns natural log of 1 plus `this` element-wise.
* `relu(): Tensor`: Returns relu of `this` element-wise.
* `sigmoid(): Tensor`: Returns sigmoid of `this` element-wise.
* `tanh(): Tensor`: Returns tanh of `this` element-wise.
* `transpose(dim1: number, dim2: number): Tensor`: Returns transposition of a tensor from two provided dimensions.
* `t(): Tensor`: Returns transposition of a 2D tensor (matrix). If `this` is not 2D, it will throw an error.
* `dot(other: TensorValue | Tensor): Tensor`: Returns the vector dot product of `this` and `other` 1D tensors (vectors). If the two are not 1D, it will throw an error.
* `mm(other: TensorValue | Tensor): Tensor`: Returns the matrix multiplication of `this` and `other` 2D tensors (matrices). If the two are not 2D, it will throw an error.
* `mv(other: TensorValue | Tensor): Tensor`: Returns the matrix multiplication of `this` 2D tensor (matrix) and `other` 1D tensor (vector). Basically if `other` is of size n, it will be reshaped into an nx1 matrix. If `this` is not 2D and `other` is not 1D, it will throw an error.
* `matmul(other: TensorValue | Tensor): Tensor`: Returns the matrix multiplication of `this` and `other`. If both are 1D then `dot` is used, if both are 2D then `mm` is used, if `this` is 2D and `other` is 1D then `mv` is used, if `this` is 1D and `other` is 2D then a size-1 dimension will be padded into `this` to do `mm`, then the padded dimension will be removed.
* `squeeze(dims?: number[] | number): Tensor`: Returns a new tensor with size-1 dims squeezed out. If `dims` is `undefined`, all size-1 dims are squeezed out. If `dims` is a number/number array, it will squeeze out dimensions at that/those positions. If a specified dimension is not size-1, it will throw an error.
* `unsqueeze(dims?: number[] | number): Tensor`: Returns a new tensor with size-1 dims pushed into specified positions. If `dims` is `undefined`, a tensor with same value, shape, and strides will be returned. If `dims` is a number/number array, it will push size-1 dimensions into that/those positions.
* `sum(dims?: number[] | number, keepDims: boolean = false): Tensor`: Returns a new tensor with axes summed. If `dims` is `undefined`, all axes will be summed into a scalar. If `dims` is a number/number array, it will sum dimensions at that/those positions. If `keepDims` is `true`, then the size-1 dimensions after summation will be kept, discarded otherwise.
* `prod(dims?: number[] | number, keepDims: boolean = false): Tensor`: Returns a new tensor with axes reduced to their products. If `dims` is `undefined`, all axes will be reduced into a scalar. If `dims` is a number/number array, it will reduce dimensions at that/those positions. If `keepDims` is `true`, then the size-1 dimensions after reduction will be kept, discarded otherwise.
* `mean(dims?: number[] | number, keepDims: boolean = false): Tensor`: Returns a new tensor with axes reduced to their means. If `dims` is `undefined`, all axes will be reduced into a scalar. If `dims` is a number/number array, it will reduce dimensions at that/those positions. If `keepDims` is `true`, then the size-1 dimensions after reduction will be kept, discarded otherwise.

Here are commonly used utilities:

* `backward()`: Calling this will recursively accumulate gradients of nodes in the DAG you have built, with whatever `this` you are calling with as the top node. Note that this will assume the gradient of the top node to be a tensor of same shape, filled with 1, and it will zero out the gradients of child nodes before calculation.
* `val(): TensorValue`: Returns the raw nD array/number form of the tensor.
* `withGrad(requiresGrad: boolean): Tensor`: Returns a copy of the tensor with requiresGrad changed and detaches from DAG (reset children, grad, gradFn, etc).
* `static fullLike(tensor: Tensor, num: number, options: TensorOptions = {}): Tensor`: Returns a new tensor of same shape and strides as `tensor`, filled with `num`, configured with `options`.
* `static onesLike(tensor: Tensor, options: TensorOptions = {}): Tensor`: Returns a new tensor of same shape and strides as `tensor`, filled with 1, configured with `options`.
* `static zerosLike(tensor: Tensor, options: TensorOptions = {}): Tensor`: Returns a new tensor of same shape and strides as `tensor`, filled with 0, configured with `options`.

Here are utilities that you probably won't have to use but they might come in handy:

* `static flatten(tensor: TensorValue): number[] | number`: Used to flatten an n-D array to 1D. If argument is a number, it would return the number.
* `static getShape(tensor: TensorValue): readonly number[]`: Used to get shape (size of each dimension) of an n-D array as a number array.
* `static getStrides(shape: readonly number[]): readonly number[]`: Used to get strides of tensor from its shape. Strides are needed internally because they are steps taken to get a value at each dimension now that the tensor has been flatten to 1D.
* `static padShape`: Used to pad shape and strides of two tensors to be of same number of dimensions.
    * args:
        * `stridesA: readonly number[]`: Strides of the first tensor.
        * `stridesB: readonly number[]`: Strides of the second tensor.
        * `shapeA: readonly number[]`: Shape of the first tensor.
        * `shapeB: readonly number[]`: Shape of the second tensor.
    * returns: A tuple of `(newStridesA, newStridesB, newShapeA, newShapeB)` with type `[readonly number[], readonly number[], readonly number[], readonly number[]]`.
* `static broadcastShapes(shapeA: readonly number[], shapeB: readonly number[]): readonly number[]`: Returns the new shape broadcasted from `shapeA` and `shapeB`. Basically if one shape's dimension is of size n, and other shape's corresponding dimension if of size n or 1, then the new shape's corresponding dimension is n, otherwise throw an error. For example `[1,2,3]` and `[4,1,3]` would be `[4,2,3]` after broadcasting.
* `static indexToCoords(index: number, shape: readonly number[], strides: readonly number[]): number[]`: Convert an index of an 1D array to coordinates (indices) of an nD array, based on the nD array's `shape` and `strides`.
* `static coordsToIndex(coords: number[], strides: readonly number[]): number`: Convert coordinates (indices) of an nD array to an index of an 1D array, based on the nD array's `strides`.
* `static coordsToUnbroadcastedIndex(coords: number[], shape: readonly number[], strides: readonly number[]): number`: Convert coordinates (indices) of an unbroadcasted nD array to an index of an 1D array, based on the nD array's `shape` and `strides`. Basically the same as above but coordinates of dimensions with size 1 are forced to be 0.
* `static shapeToSize(shape: readonly number[]): number`: Convert shape into 1D array size.
* `static elementWiseAB(tA: Tensor, tB: Tensor, op: (tA: number, tB: number) => number): Tensor`: Perform a custom element-wise `op` between two tensors, returns a new tensor that holds the result.
* `static elementWiseSelf(tA: Tensor, op: (tA: number) => number): Tensor`: Perform a custom element-wise `op` on a tensor, returns a new tensor that holds the result.
* `elementWiseABDAG`: Perform a custom element-wise op between this tensor with another tensor. If `this` or `other` have `requiresGrad` as `true`, it will build a DAG node for future gradient computation.
    * args:
        * `other: TensorValue | Tensor`: The other tensor.
        * `op: (a: number, b: number) => number`: The custom op to do element-wise.
        * `thisGrad: (self: Tensor, other: Tensor, outGrad: Tensor) => Tensor = () => new Tensor(0)`: Custom gradient for `this` tensor if `this.requiresGrad` is `true`, returns a tensor that will be assigned to `this.grad`. Note that `self` represents `this` tensor, `other` represents the `other` tensor above, and `outGrad` represents the upstream gradient, but all of these tensors have all gradient-related operations disabled and are not the original tensors.
        * `otherGrad: (self: Tensor, other: Tensor, outGrad: Tensor) => Tensor = () => new Tensor(0)`: Same as above but assigned to `other.grad`.
    * returns: A new `Tensor`. If `this.requiresGrad` is `true`, then `this` will be a child of the new tensor, same with `other`.
* `elementWiseSelfDAG`: Perform a custom element-wise op on a tensor. If `this.requiresGrad` is `true`, it will build a DAG node for future gradient computation.
    * args:
        * `op: (a: number) => number`: The custom op to do element-wise.
        * `thisGrad: (self: Tensor, outGrad: Tensor) => Tensor = () => new Tensor(0)`: Custom gradient for `this` tensor if `this.requiresGrad` is `true`, returns a tensor that will be assigned to `this.grad`. Note that `self` represents `this` tensor, and `outGrad` represents the upstream gradient, but all of these tensors have all gradient-related operations disabled and are not the original tensors.
    * returns: A new `Tensor`. If `this.requiresGrad` is `true`, then `this` will be a child of the new tensor`.
* `static forceTensor(value: TensorValue | Tensor): Tensor`: Returns the argument if it already is a `Tensor` instance, otherwise create a new `Tensor` instance with `value` as input.
* `static addGrad(tensor: Tensor, accumGrad: Tensor)`: Add to the `grad` prop of a tensor. It can handle broadcasted shapes and make `accumGrad` fit `tensor`'s shape.


# Examples

## Simple quadratic function

```js
const { Tensor } = require("catniff");

// Create a new scalar 2
const x = new Tensor(2, { requiresGrad: true });
// Calculate x^2 + x and build a DAG
const L = x.pow(2).add(x);

// Accumulate gradients of all nodes
L.backward();

// Log out raw values of x's gradient and L's gradient
console.log(x.grad.val(), L.grad.val());
```

## Xornet

Here is an MLP implemented to do the XOR operation:

```js
const { Tensor } = require("catniff"), rand = () => Math.random() * 2 - 1;

class Xornet {
    constructor(options = {}) {
        // 2->2->1 xornet
        this.w1 = new Tensor(options.w1 || [
            [rand(), rand()],
            [rand(), rand()]
        ], { requiresGrad: true });
        this.b1 = new Tensor(options.b1 || [0, 0], { requiresGrad: true });
        this.w2 = new Tensor(options.w2 || [
            [rand()],
            [rand()]
        ], { requiresGrad: true });
        this.b2 = new Tensor(options.b2 || [0], { requiresGrad: true });
        this.lr = options.lr || 0.5;
    }

    forward(input) {
        return new Tensor(input)
                    .matmul(this.w1)
                    .add(this.b1)
                    .sigmoid()
                    .matmul(this.w2)
                    .add(this.b2)
                    .sigmoid();
    }

    backprop(input, target) {
        const T = new Tensor(target);
        const Y = this.forward(input);
        const L = Y.sub(T).pow(2).mul(0.5);

        L.backward();

        // We disable gradient collecting first to calculate new weight, then enable it for next pass
        this.w1 = this.w1.withGrad(false).sub(this.w1.grad.mul(this.lr)).withGrad(true);
        this.w2 = this.w2.withGrad(false).sub(this.w2.grad.mul(this.lr)).withGrad(true);
        this.b1 = this.b1.withGrad(false).sub(this.b1.grad.mul(this.lr)).withGrad(true);
        this.b2 = this.b2.withGrad(false).sub(this.b2.grad.mul(this.lr)).withGrad(true);
    }
}

const xornet = new Xornet();

for (let epoch = 0; epoch < 30000; epoch++) {
    xornet.backprop([1,0], [1]);
    xornet.backprop([0,1], [1]);
    xornet.backprop([0,0], [0]);
    xornet.backprop([1,1], [0]);
}

console.log(xornet.forward([1,1]).val()); // 0-ish
console.log(xornet.forward([1,0]).val()); // 1-ish
console.log(xornet.forward([0,1]).val()); // 1-ish
console.log(xornet.forward([0,0]).val()); // 0-ish
```

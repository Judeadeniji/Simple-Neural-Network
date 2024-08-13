class Matrix {
  data: number[][];
  shape: [number, number];
  constructor(shape?: [number, number], init?: number[][]) {
    if (init) {
      this.data = init;
      this.shape = [init.length, init[0].length];
    } else {
      this.data = Array.from(Array(shape![0]), () =>
        new Array(shape![1]).fill(0)
      );
      this.shape = [...shape!];
    }
  }

  multiply(matrix: number | Matrix) {
    if (matrix instanceof Matrix) {
      if (this.shape[1] !== matrix.shape[0]) {
        // console.log(matrix.constructor === this.constructor, this.data);
        Matrix.error(
          `Can not multiply this two matrices. the object: ${JSON.stringify(
            this.shape
          )} multiplied by: ${JSON.stringify(matrix.shape)}`
        );
      }

      const newMatrix = new Matrix([this.shape[0], matrix.shape[1]]);

      for (let i = 0; i < newMatrix.shape[0]; i++) {
        for (let j = 0; j < newMatrix.shape[i]; j++) {
          let sum = 0;
          for (let k = 0; k < this.shape[1]; k++) {
            sum += this.data[i][k] * matrix.data[k][j];
          }
          newMatrix.data[i][j] = sum;
        }
      }
      return newMatrix;
    }

    const newMatrix = new Matrix(this.shape);

    for (let i = 0; i < newMatrix.shape[0]; i++) {
      for (let j = 0; j < newMatrix.shape[i]; j++) {
        let sum = 0;
        for (let k = 0; k < this.shape[1]; k++) {
          sum += this.data[i][k] * matrix;
        }
        newMatrix.data[i][j] = sum;
      }
    }
    return newMatrix;
  }

  add(matrix: number | Matrix) {
    if (matrix instanceof Matrix) {
      let newMatrix = new Matrix(this.shape);

      if (
        this.shape[0] === matrix.shape[0] &&
        this.shape[1] === matrix.shape[1]
      ) {
        // Element-wise addition for matrices of the same shape
        for (let i = 0; i < this.shape[0]; i++) {
          for (let j = 0; j < this.shape[1]; j++) {
            newMatrix.data[i][j] = this.data[i][j] + matrix.data[i][j];
          }
        }
      } else if (matrix.shape[0] === 1 && matrix.shape[1] === this.shape[1]) {
        // Broadcasting for row vector
        for (let i = 0; i < this.shape[0]; i++) {
          for (let j = 0; j < this.shape[1]; j++) {
            newMatrix.data[i][j] = this.data[i][j] + matrix.data[0][j];
          }
        }
      } else if (matrix.shape[1] === 1 && matrix.shape[0] === this.shape[0]) {
        // Broadcasting for column vector
        for (let i = 0; i < this.shape[0]; i++) {
          for (let j = 0; j < this.shape[1]; j++) {
            newMatrix.data[i][j] = this.data[i][j] + matrix.data[i][0];
          }
        }
      } else {
        console.log(matrix.shape, this.shape);
        throw new Error("Matrix dimensions are not compatible for addition");
      }

      return newMatrix;
    } else if (typeof matrix === "number") {
      // Scalar addition
      return this.map((value) => value + matrix);
    } else {
      throw new Error("Invalid input type for addition");
    }
  }

  subtract(matrix: Matrix | number) {
    const newMatrix = new Matrix([this.shape[0], this.shape[1]]);

    if (typeof matrix === "number") {
      // Subtract a scalar from each element of the matrix
      for (let i = 0; i < this.shape[0]; i++) {
        for (let j = 0; j < this.shape[1]; j++) {
          newMatrix.data[i][j] = this.data[i][j] - matrix;
        }
      }
    } else {
      // Ensure the matrices have the same dimensions
      if (
        this.shape[0] !== matrix.shape[0] ||
        this.shape[1] !== matrix.shape[1]
      ) {
        // throw new Error("Matrix dimensions must match for subtraction.");
      }

      // Subtract element-wise
      for (let i = 0; i < this.shape[0]; i++) {
        for (let j = 0; j < this.shape[1]; j++) {
          newMatrix.data[i][j] = this.data[i][j] - matrix.data[i][j];
        }
      }
    }

    return newMatrix;
  }

  map(func: (v: number, i: number, j: number) => number) {
    const newMatrix = new Matrix(this.shape);
    // Apply a function to every element of matrix
    for (let i = 0; i < this.shape[0]; i++) {
      for (let j = 0; j < this.shape[1]; j++) {
        let val = this.data[i][j];
        newMatrix.data[i][j] = func(val, i, j);
      }
    }

    return newMatrix;
  }

  reduce<U>(
    callback: (accumulator: U, value: number, row: number, col: number) => U,
    initialValue: U
  ): U {
    let accumulator = initialValue;

    for (let i = 0; i < this.shape[0]; i++) {
      for (let j = 0; j < this.shape[1]; j++) {
        accumulator = callback(accumulator, this.data[i][j], i, j);
      }
    }

    return accumulator;
  }

  randomize() {
    for (let i = 0; i < this.shape[0]; i++)
      for (let j = 0; j < this.shape[1]; j++)
        this.data[i][j] = Math.random() * 2 - 1; //between -1 and 1

    return this;
  }

  transpose(): Matrix {
    const newMatrix = new Matrix([this.shape[1], this.shape[0]]);
    const transposedData: number[][] = [];
    for (let j = 0; j < this.shape[1]; j++) {
      transposedData.push([]);
      for (let i = 0; i < this.shape[0]; i++) {
        transposedData[j].push(this.data[i][j]);
      }
    }

    newMatrix.data = transposedData;
    return newMatrix;
  }

  static error(message: string): never {
    const e = new Error(message);
    e.name = "[Matrix Error]";

    throw e;
  }

  static from(numberArray: number[][]) {
    let newMatrix = new Matrix([numberArray.length, numberArray[0].length]);
    newMatrix = newMatrix.map((_, i, j) => numberArray[i][j]);

    return newMatrix;
  }

  static dot(A: number[][], B: number[][]): number[][] {
    const rowsA = A.length;
    const colsA = A[0].length;
    const colsB = B[0].length;

    if (colsA !== B.length) {
      Matrix.error(
        `Can not multiply this two matrices. the object: ${JSON.stringify([
          rowsA,
          colsA,
        ])} multiplied by: ${JSON.stringify([B.length, colsB])}`
      );
    }

    const result: number[][] = Array.from({ length: rowsA }, () =>
      Array(colsB).fill(0)
    );

    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsB; j++) {
        for (let k = 0; k < colsA; k++) {
          result[i][j] += A[i][k] * B[k][j];
        }
      }
    }

    return result;
  }

  static add(A: number[][], B: number[][]): number[][] {
    return A.map((row, i) => row.map((val, j) => val + B[i][j]));
  }

  static scalarMultiply(A: number[][], scalar: number): number[][] {
    return A.map((row) => row.map((val) => val * scalar));
  }

  static applyFunction(A: number[][], func: (x: number) => number): number[][] {
    return A.map((row) => row.map((val) => func(val)));
  }
  static subtract(A: number[][], B: number[][]): number[][] {
    return A.map((row, i) => row.map((val, j) => val - B[i][j]));
  }

  static transpose(A: number[][]): number[][] {
    return A[0].map((_, i) => A.map((row) => row[i]));
  }

  static multiplyElementWise(A: number[][], B: number[][]): number[][] {
    return A.map((row, i) => row.map((val, j) => val * B[i][j]));
  }
}

export { Matrix };

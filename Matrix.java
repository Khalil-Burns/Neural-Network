public class Matrix {

    double[][] values;
    boolean vector;

    public Matrix(int w, int h, boolean createNormal) {
        this.values = new double[w][h]; //I will be flipping x and y to make it more intuitive (i.e. it will be [x][y] instead of [y][x])
        if (createNormal) {
            for (int i = 0; i < w; i++) {
                for (int j = 0; j < h; j++) {
                    this.values[i][j] = Network.randn();
                }
            }
        }
        if (h == 1) {
            this.vector = true;
        }
        else {
            this.vector = false;
        }
    }
    public Matrix(double[][] vals) {
        this.values = vals;
        if (vals[0].length == 1) {
            this.vector = true;
        }
        else {
            this.vector = false;
        }
    }
    public Matrix(Matrix m) {
        this.values = m.values;
        this.vector = m.vector;
    }

    public Matrix add(Matrix m) {
        int n, m1, m2, p;
        n = this.values.length;
        m1 = this.values[0].length;
        m2 = m.values.length;
        p = m.values[0].length;
        if (n != m2) {
            System.out.println("ERROR: Matrix -> Matrix add(Matrix m), matrix rows do not match");
            return(null);
        }
        if (m1 != p) {
            System.out.println("ERROR: Matrix -> Matrix add(Matrix m), matrix columns do not match");
            return(null);
        }

        Matrix res = new Matrix(n, m1, false);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m1; j++) {
                res.values[i][j] = this.values[i][j] + m.values[i][j];
            }
        }
        return(res);
    }
    public Matrix subtract(Matrix m) {
        int n, m1, m2, p;
        n = this.values.length;
        m1 = this.values[0].length;
        m2 = m.values.length;
        p = m.values[0].length;
        if (n != m2) {
            System.out.println("ERROR: Matrix -> Matrix subtract(Matrix m), matrix rows do not match");
            return(null);
        }
        if (m1 != p) {
            System.out.println("ERROR: Matrix -> Matrix subtract(Matrix m), matrix columns do not match");
            return(null);
        }

        Matrix res = new Matrix(n, m1, false);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m1; j++) {
                res.values[i][j] = this.values[i][j] - m.values[i][j];
            }
        }
        return(res);
    }
    public Matrix multiply(Matrix m) { //with this structure, I would need to rotate the first vector if I'm multiplying two vectors
        int n, m1, m2, p;
        n = this.values.length;
        m1 = this.values[0].length;
        m2 = m.values.length;
        p = m.values[0].length;
        if (this.vector && m.vector) {
            if (m1 != p) {
                System.out.println("ERROR: Matrix -> Matrix multiply(Matrix m), matrix sizes do not match");
                return(null);
            }
            Matrix res = new Matrix(1, 1, false);
            res.values[0][0] = 0;
            for (int i = 0; i < n; i++) {
                res.values[0][0] += this.values[i][0] * m.values[i][0];
            }
            return(res);
        }
        if (m1 != m2) {
            System.out.println("ERROR: Matrix -> Matrix multiply(Matrix m), matrix sizes do not match");
            return(null);
        }

        Matrix res = new Matrix(n, p, false);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                res.values[i][j] = 0;
                for (int k = 0; k < m1; k++) {
                    res.values[i][j] += this.values[i][k] * m.values[k][j];
                }
            }
        }
        return(res);
    }
    public Matrix hadamard(Matrix m) { //with this structure, I would need to rotate the first vector if I'm multiplying two vectors
        int n, m1, m2, p;
        n = this.values.length;
        m1 = this.values[0].length;
        m2 = m.values.length;
        p = m.values[0].length;
        if (this.vector && m.vector) {
            if (n != m2) {
                System.out.println("ERROR: Matrix -> Matrix multiply(Matrix m), matrix sizes do not match");
                return(null);
            }
            Matrix res = new Matrix(this.values.length, 1, false);
            for (int i = 0; i < n; i++) {
                res.values[i][0] = this.values[i][0] * m.values[i][0];
            }
            return(res);
        }
        if (m1 != m2) {
            System.out.println("ERROR: Matrix -> Matrix multiply(Matrix m), matrix sizes do not match");
            return(null);
        }

        Matrix res = new Matrix(n, p, false);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                res.values[i][j] = 0;
                for (int k = 0; k < m1; k++) {
                    res.values[i][j] += this.values[i][k] * m.values[k][j];
                }
            }
        }
        return(res);
    }
    public static Matrix multiply(Matrix a, Matrix b) {
		Matrix temp = new Matrix(a.values.length, b.values[0].length, false);
		for(int i=0;i<temp.values.length;i++)
		{
			for(int j=0;j<temp.values[i].length;j++)
			{
				double sum=0;
				for(int k=0;k<a.values[0].length;k++)
				{
					sum += a.values[i][k]*b.values[k][j];
				}
				temp.values[i][j]=sum;
			}
		}
		return temp;
	}
    public Matrix transpose() {
        Matrix res = new Matrix(this.values[0].length, this.values.length, false);
        for (int i = 0; i < res.values.length; i++) {
            for (int j = 0; j < res.values[i].length; j++) {
                res.values[i][j] = this.values[j][i];
            }
        }
        return(res);
    }
    public Matrix scale(double x) {
        Matrix res = new Matrix(this);
        for (int i = 0; i < res.values.length; i++) {
            for (int j = 0; j < res.values[i].length; j++) {
                res.values[i][j] = this.values[i][j] * x;
            }
        }
        return(res);
    }
    public Matrix sigmoid() {
        Matrix res = new Matrix(this);
        for (int i = 0; i < res.values.length; i++) {
            for (int j = 0; j < res.values[i].length; j++) {
                res.values[i][j] = Network.sigmoid(this.values[i][j]);
            }
        }
        return(res);
    }
    public Matrix dSigmoid() {
        Matrix res = new Matrix(this);
        for (int i = 0; i < res.values.length; i++) {
            for (int j = 0; j < res.values[i].length; j++) {
                res.values[i][j] = Network.dSigmoid(this.values[i][j]);
            }
        }
        return(res);
    }
    public Matrix relu() {
        Matrix res = new Matrix(this);
        for (int i = 0; i < res.values.length; i++) {
            for (int j = 0; j < res.values[i].length; j++) {
                res.values[i][j] = Network.relu(this.values[i][j]);
            }
        }
        return(res);
    }

    @Override
	public String toString() {
        String res = "[";
        for (int i = 0; i < this.values.length; i++) {
            String s = "[";
            for (int j = 0; j < this.values[i].length; j++) {
                if (j != this.values[i].length - 1) {
                    s += (this.values[i][j] + ", ");
                }
                else {
                    s += (this.values[i][j] + "");
                }
            }
            if (i == this.values.length - 1) {
                s += "]";
            }
            else {
                s += "],\n";
            }
            res += s;
        }
        res += "]\n";
		return(res);
	}
}

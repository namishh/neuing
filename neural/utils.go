package neural

import "math"

// transposition of a matrix ; rotating a matrix by 90 degrees
// https://en.wikipedia.org/wiki/Transpose
func transpose(m [][]float64) [][]float64 {
	if len(m) == 0 {
		return m
	}
	r := len(m)
	c := len(m[0])
	t := make([][]float64, c)
	for i := 0; i < c; i++ {
		t[i] = make([]float64, r)
		for j := 0; j < r; j++ {
			t[i][j] = m[j][i]
		}
	}
	return t
}

// standard product of two matrices
func matrixProduct(m1, m2 [][]float64) [][]float64 {
	r1 := len(m1)
	c1 := len(m1[0])
	r2 := len(m2)
	c2 := len(m2[0])

	if c1 != r2 {
		panic("Matrices can't be multiplied")
	}

	result := make([][]float64, r1)
	for i := 0; i < r1; i++ {
		result[i] = make([]float64, c2)
		for j := 0; j < c2; j++ {
			for k := 0; k < c1; k++ {
				result[i][j] += m1[i][k] * m2[k][j]
			}
		}
	}
	return result
}

// implmentation of numpy.zeros
func zeros(x int, y int) [][]float64 {
	grid := make([][]float64, x)
	for i := 0; i < x; i++ {
		grid[i] = make([]float64, y)
		for j := 0; j < y; j++ {
			grid[i][j] = 0
		}
	}
	return grid
}

// implmentation of numpy.clip
func Clip(x float64, min float64, max float64) float64 {
	if x < min {
		return min
	} else if x > max {
		return max
	}

	return x
}

// get largest emelent of an array
func MaxElement(arr []float64) float64 {
	max := arr[0]
	for _, val := range arr {
		if val > max {
			max = val
		}
	}
	return max
}

// small helper utility
func IndexOf(element float64, data []float64) int {
	for k, v := range data {
		if element == v {
			return k
		}
	}
	return -1 //not found.
}

func addBias(p [][]float64, bias []float64) (output [][]float64) {
	l := len(p)
	if l == 0 {
		return nil
	}
	if l != len(bias) {
		panic("Can't Add Bias: Number of biases must match the number of slices in p")
	}
	output = make([][]float64, l)
	for i := 0; i < l; i++ {
		output[i] = make([]float64, len(p[i]))
		for j := 0; j < len(p[i]); j++ {
			output[i][j] = p[i][j] + bias[i]
		}
	}
	return
}

func getOutput(inputs [][]float64, weights [][]float64, bias []float64) [][]float64 {
	return addBias(matrixProduct(inputs, transpose(weights)), bias)
}

// replace of numpy.exp
func Exponential(inputs [][]float64) [][]float64 {
	E := math.E
	for i, arr := range inputs {
		for j, val := range arr {
			inputs[i][j] = math.Pow(E, val)
		}
	}
	return inputs
}

func Max(inputs [][]float64) [][]float64 {
	result := make([][]float64, len(inputs))
	for i, row := range inputs {
		maxValue := math.Inf(-1) // negative infinity
		for _, value := range row {
			if value > maxValue {
				maxValue = value
			}
		}
		result[i] = []float64{maxValue}
	}
	return result
}

func Accuracy(actOutput [][]float64, y []float64) float64 {
	predictions := make([]float64, len(actOutput))
	for i, arr := range actOutput {
		predictions[i] = float64(IndexOf(MaxElement(arr), arr))
	}
	numCorrect := 0.0
	for i := 0; i < len(predictions); i++ {
		if predictions[i] == y[i] {
			numCorrect++
		}
	}
	accuracy := numCorrect / float64(len(predictions))
	return accuracy
}

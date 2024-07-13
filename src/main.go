package main

import (
	"fmt"
	"math"
	"math/rand"
)

// this is just generated from chatgpt which converts
// https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py this code to this
// orignal license : https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
func generateData(samples, classes int) (X [][]float64, y []int) {
	X = make([][]float64, samples*classes)
	y = make([]int, samples*classes)
	for classNumber := 0; classNumber < classes; classNumber++ {
		ixStart := samples * classNumber
		r := make([]float64, samples)
		t := make([]float64, samples)
		for i := 0; i < samples; i++ {
			r[i] = float64(i) / float64(samples-1)
			t[i] = float64(classNumber)*4.0 + float64(classNumber+1)*4.0*float64(i)/float64(samples) + rand.NormFloat64()*0.2
			X[ixStart+i] = []float64{r[i] * math.Sin(t[i]*2.5), r[i] * math.Cos(t[i]*2.5)}
			y[ixStart+i] = classNumber
		}
	}
	return
}

func transpose(m [][]int) [][]int {
	if len(m) == 0 {
		return m
	}
	r := len(m)
	c := len(m[0])
	t := make([][]int, c)
	for i := 0; i < c; i++ {
		t[i] = make([]int, r)
		for j := 0; j < r; j++ {
			t[i][j] = m[j][i]
		}
	}
	return t
}

func matrixProduct(m1, m2 [][]int) [][]int {
	r1 := len(m1)
	c1 := len(m1[0])
	r2 := len(m2)
	c2 := len(m2[0])

	if c1 != r2 {
		panic("Matrices can't be multiplied")
	}

	result := make([][]int, r1)
	for i := 0; i < r1; i++ {
		result[i] = make([]int, c2)
		for j := 0; j < c2; j++ {
			for k := 0; k < c1; k++ {
				result[i][j] += m1[i][k] * m2[k][j]
			}
		}
	}
	return result
}

func addBias(p [][]int, bias []int) (output [][]int) {
	l := len(p)
	if l == 0 {
		return nil
	}
	if l != len(bias) {
		panic("Can't Add Bias: Number of biases must match the number of slices in p")
	}
	output = make([][]int, l)
	for i := 0; i < l; i++ {
		output[i] = make([]int, len(p[i]))
		for j := 0; j < len(p[i]); j++ {
			output[i][j] = p[i][j] + bias[i]
		}
	}
	return
}

func getOutput(inputs [][]int, weights [][]int, bias []int) [][]int {
	return addBias(matrixProduct(inputs, transpose(weights)), bias)
}

func main() {
	fmt.Println("Hello World!")
	inputs := [][]int{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	weights := [][]int{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}

	bias := []int{1, 2, 3}

	X, y := generateData(100, 3)
	fmt.Println(X, y)

	fmt.Println(getOutput(inputs, weights, bias))
}

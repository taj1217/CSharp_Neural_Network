using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        //nodes
        private int inodes;
        private int hnodes;
        private int onodes;

        //weights
        private double[,] wih;
        private double[,] who;

        //Learning Rate
        private double lr;

        //Initialization
        public NeuralNetwork(int InputNodes, int HiddenNodes, int OutputNodes, double LearningRate)
        {
            inodes = InputNodes;
            hnodes = HiddenNodes;
            onodes = OutputNodes;

            Random ran = new Random();
            wih = new double[HiddenNodes,InputNodes];
            for(int i = 0;  i< HiddenNodes; i++)
            {
                for (int j = 0; j < InputNodes; j++)
                {
                    wih[i, j] = (ran.NextDouble() - .5);
                }
            }

            who = new double[OutputNodes, HiddenNodes];
            for (int i = 0; i < OutputNodes; i++)
            {
                for (int j = 0; j < HiddenNodes; j++)
                {
                    who[i, j] = (ran.NextDouble() - .5);
                }
            }

            lr = LearningRate;
        }

        public void ToString()
        {
            Console.WriteLine("Input Nodes = " + this.inodes);
            Console.WriteLine("Hidden Nodes = " + this.hnodes);
            Console.WriteLine("Output Nodes = " + this.onodes);
            Console.WriteLine("Wih");
            for (int i = 0; i < this.wih.GetLength(0); i++) { for (int j = 0; j < this.wih.GetLength(1); j++) { Console.Write(" " + this.wih[i, j]); } Console.WriteLine(); }
            Console.WriteLine("Who");
            for (int i = 0; i < this.who.GetLength(0); i++) { for (int j = 0; j < this.who.GetLength(1); j++) { Console.Write(" " + this.who[i, j]); } Console.WriteLine(); }
            Console.WriteLine("Learning Rate = " + this.lr);
        }

        public void Train(double[] inputs, double[] targets)
        {
            //Querry 
            double[] hiddenin = Mmul(this.wih, inputs);
            double[] hiddenout = Sigmoid(hiddenin);
            double[] finalin = Mmul(this.who, hiddenout);
            double[] finalout = Sigmoid(finalin);
            double[] output = finalout;
            for(int i = 0; i< output.Length; i++)
            {
                //Console.WriteLine("Input: "+inputs[i]);
                //Console.WriteLine("Output: "+ output[i]);
                //Console.WriteLine("Actual: " + Math.Sin(inputs[i]));
                Console.WriteLine("Difference: " + (output[i] - Math.Sin(inputs[i])));
            }
            

            double[] output_error = new double[targets.Length];
            for (int i = 0; i < targets.Length; i++) { output_error[i] = targets[i] - output[i];}
            double[] hidden_error = Mmul(Transpose(who), output_error);

            //Calculate new WHO
            double[] finalerror = new double[targets.Length];
            for(int i = 0; i<targets.Length; i++) { finalerror[i] = (output_error[i] * output[i] * (1 - output[i])); }
            double [,] FHidden_Errors = Mmul(finalerror, hiddenout);
            who = Aadd(who, FHidden_Errors);

            //Calculate new WIH
            double[] middleerror = new double[hidden_error.Length];
            for (int i = 0; i < targets.Length; i++) { middleerror[i] = (hidden_error[i] * hiddenout[i] * (1 - hiddenout[i])); }
            double[,] FInput_Errors = Mmul(middleerror, inputs);
            wih = Aadd(wih, FInput_Errors);
        }

        public double[] query(double[] inputs)
        {
            double[] hiddenin = Mmul(this.wih, inputs);
            double[] hiddenout = Sigmoid(hiddenin);
            Console.WriteLine("Finished Input");
            double[] finalin = Mmul(this.who, hiddenout);
            double[] finalout = Sigmoid(finalin);
            Console.WriteLine("Finished Output");
            return finalout;
        }

        public static double[,] Transpose(double[,] input)
        {
            double[,] output = new double[input.GetLength(1), input.GetLength(0)];
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    output[j, i] = input[i, j];
                }
            }
            return output;
        }

        public static double[] Sigmoid(double[] input)
        {
            double[] output = new double[input.Length];
            for(int i = 0; i < input.Length; i++)
            {
                output[i] = (1 / (1 + (Math.Pow(Math.E, -input[i]))));
            }

            return output;
        }

        public static double[] Mmul(double[,] weights, double[] inputs)
        {
            double[] output = new double[weights.GetLength(0)];
            if (weights.GetLength(1) != inputs.Length) { Console.WriteLine("Incorrect Matrix Size!"); return output; }
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {

                    output[i] += (weights[i, j] * inputs[j]);
                }
            }

            return output;
        }

        public double[,] Mmul(double[] weights, double[] inputs)
        {
        double[,] output = new double[weights.Length, inputs.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            for (int j = 0; j < inputs.Length; j++)
            {
                output[i, j] = (weights[i] * inputs[j])*this.lr;
            }
        }
        return output;
        }

        public static double[,] Aadd(double[,] array, double[,] errors)
        {
            double[,] output = new double[array.GetLength(0), array.GetLength(1)];
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    output[i, j] = array[i, j] + errors[i, j];
                }
            }
            return output;
        }
    }
}

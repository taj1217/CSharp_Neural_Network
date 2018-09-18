using System;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            Random ran = new Random();
            NeuralNetwork n = new NeuralNetwork(1, 100, 1, .03);
            double[] input = new double[1];
            double[] output = new double[1];
            for (int i = 0; i < 3000; i++)
            {
                for (int j = 0; j < 1; j++)
                {
                    double number = ran.NextDouble();
                    input[j] = number;
                    output[j] = Math.Sin(number);
                }
                n.Train(input, output);
            }
            Console.WriteLine("Input: ");
            double[] testinput = new double[1];
            testinput[0] = Convert.ToDouble(Console.ReadLine());
            Console.WriteLine(testinput[0]);

            double[] testgoal = testinput;
            for(int i = 0; i < testgoal.Length; i++) { testgoal[i] = Math.Sin(testinput[i]); }
            double[] testoutput = new double[10];
            testoutput = n.query(testinput);
            for(int i = 0; i < testoutput.Length; i++) { Console.WriteLine(testoutput[i]+" " + testgoal[i]+" " + (output[i]-testgoal[i])); }
            Console.ReadLine();
        }
    }
}

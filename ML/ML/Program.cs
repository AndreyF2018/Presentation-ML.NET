using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
namespace ML
{
    class Program
    {
        static void Main(string[] args)
        {
            string[] lines = File.ReadAllLines("Car.txt");
            double[,] data = new double[lines.Length - 1, lines[0].Split(' ').Length];
            for (int i = 1; i < lines.Length - 1; i++)
            {
                string[] temp = lines[i].Split(' ');
                for (int j = 0; j < temp.Length; j++)
                {
                    data[i, j] = Convert.ToDouble(temp[j]);
                }
            }

            MLContext mlContext = new MLContext();

            List<CarPriceData> carPriceDataList = new List<CarPriceData>();
            for (int i = 1; i < data.GetLength(0); i++)
            {
                carPriceDataList.Add(new CarPriceData() { CarPrice = data[i, 0], Income = data[i, 1] });
            }

            CarPriceData[] carPriceData = carPriceDataList.ToArray();

            IDataView fittingData = mlContext.Data.LoadFromEnumerable(carPriceData);

            var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Income" })
             .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "CarPrice", maximumNumberOfIterations: 100));

            var model = pipeline.Fit(fittingData);

            var IncomeForPrediction = 14740390;
            var CarPricePrediction = mlContext.Model.CreatePredictionEngine<CarPriceData, Prediction>(model).Predict(IncomeForPrediction);

            Console.WriteLine("Предсказанная стоимость машины для заданного дохода = " + CarPricePrediction);
            mlContext.Model.Save(model, fittingData.Schema, "model.zip");

            DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(fittingData, testFraction: 0.2);
            IDataView trainData = dataSplit.TrainSet;
            IDataView testData = dataSplit.TestSet;
        }
    }
}

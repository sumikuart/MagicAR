using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV;
using MagicAR.Code.BorrowedCode;

namespace MagicAR.Code
{
    internal static class GameShapes
    {

        public static void DrawTriangle(IInputOutputArray img, Matrix<float> projection, MCvScalar ShapeColor, float scale = 1)
        {
            Matrix<float>[] worldPoints = new[]
            {
                new Matrix<float>(new float[] { 0, 0, 0, 1 }), new Matrix<float>(new float[] { scale, 0, 0, 1 }),
                new Matrix<float>(new float[] { scale/2, scale, 0, 1 }), new Matrix<float>(new float[] { scale/2, scale/2, -scale * 0.7f, 1 }),
            };

            Tuple<int, int>[] lineIndexes = new[] {
                Tuple.Create(0, 3), Tuple.Create(1, 3), // Floor
                Tuple.Create(2, 3), Tuple.Create(0, 1),
                Tuple.Create(0, 2), Tuple.Create(1, 2), // Top
            };

            Point[] screenPoints = worldPoints
                 .Select(x => UtilityAR.WorldToScreen(x, projection)).ToArray();

            // Draw filled floor
            VectorOfVectorOfPoint floorContourX = new VectorOfVectorOfPoint(new VectorOfPoint(new Point[]{ screenPoints[0], screenPoints[1], screenPoints[3]}));
            CvInvoke.DrawContours(img, floorContourX, -1, ShapeColor, -3);

            // Draw filled floor
            VectorOfVectorOfPoint floorContourY = new VectorOfVectorOfPoint(new VectorOfPoint(new Point[]{ screenPoints[1], screenPoints[2], screenPoints[3] }));
            CvInvoke.DrawContours(img, floorContourY, -1, ShapeColor, -3);

            // Draw filled floor
            VectorOfVectorOfPoint floorContourZ = new VectorOfVectorOfPoint(new VectorOfPoint(new Point[] { screenPoints[2], screenPoints[0], screenPoints[3] }));
            CvInvoke.DrawContours(img, floorContourZ, -1, ShapeColor, -3);


  

            // Draw pillars
            foreach (Tuple<int, int> li in lineIndexes)
            {
                Point p1 = screenPoints[li.Item1];
                Point p2 = screenPoints[li.Item2];

                CvInvoke.Line(img, p1, p2, new MCvScalar(0, 0, 255), 1);
            
            }
        }


        public static void DrawCube(IInputOutputArray img, Matrix<float> projection, MCvScalar ShapeColor, float scale = 1)
        {
            Matrix<float>[] worldPoints = new[]
           {
                new Matrix<float>(new float[] { 0, 0, 0, 1 }), new Matrix<float>(new float[] { scale, 0, 0, 1 }),
                new Matrix<float>(new float[] { scale, scale, 0, 1 }), new Matrix<float>(new float[] { 0, scale, 0, 1 }),
                new Matrix<float>(new float[] { 0, 0, -scale, 1 }), new Matrix<float>(new float[] { scale, 0, -scale, 1 }),
                new Matrix<float>(new float[] { scale, scale, -scale, 1 }), new Matrix<float>(new float[] { 0, scale, -scale, 1 })
            };

            Point[] screenPoints = worldPoints
                .Select(x => UtilityAR.WorldToScreen(x, projection)).ToArray();

            Tuple<int, int>[] lineIndexes = new[] {
                Tuple.Create(0, 1), Tuple.Create(1, 2), // Floor
                Tuple.Create(2, 3), Tuple.Create(3, 0),
                Tuple.Create(4, 5), Tuple.Create(5, 6), // Top
                Tuple.Create(6, 7), Tuple.Create(7, 4),
                Tuple.Create(0, 4), Tuple.Create(1, 5), // Pillars
                Tuple.Create(2, 6), Tuple.Create(3, 7)
            };

            // Draw filled floor
            VectorOfVectorOfPoint floorContourA = new VectorOfVectorOfPoint(new VectorOfPoint(new Point[] { screenPoints[0], screenPoints[1], screenPoints[5], screenPoints[4] }));
            CvInvoke.DrawContours(img, floorContourA, -1, ShapeColor, -3);

            VectorOfVectorOfPoint floorContourB = new VectorOfVectorOfPoint(new VectorOfPoint(new Point[] { screenPoints[1], screenPoints[2], screenPoints[6], screenPoints[5] }));
            CvInvoke.DrawContours(img, floorContourB, -1, ShapeColor, -3);
     
            VectorOfVectorOfPoint floorContourC = new VectorOfVectorOfPoint(new VectorOfPoint(new Point[] { screenPoints[2], screenPoints[3], screenPoints[7], screenPoints[6] }));
            CvInvoke.DrawContours(img, floorContourC, -1, ShapeColor, -3);
 
            VectorOfVectorOfPoint floorContourD = new VectorOfVectorOfPoint(new VectorOfPoint(new Point[] { screenPoints[3], screenPoints[0], screenPoints[4], screenPoints[7] }));
            CvInvoke.DrawContours(img, floorContourD, -1, ShapeColor, -3);

            VectorOfVectorOfPoint floorContourE = new VectorOfVectorOfPoint(new VectorOfPoint(new Point[] { screenPoints[4], screenPoints[5], screenPoints[6], screenPoints[7] }));
            CvInvoke.DrawContours(img, floorContourE, -1, ShapeColor, -3);

            // Draw pillars
            foreach (Tuple<int, int> li in lineIndexes)
            {
                Point p1 = screenPoints[li.Item1];
                Point p2 = screenPoints[li.Item2];

                CvInvoke.Line(img, p1, p2, new MCvScalar(0, 0, 255), 1);
            }


        }


    }
}

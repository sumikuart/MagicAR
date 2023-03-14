using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using MagicAR.Code.BorrowedCode;
using System.Windows.Input;
using System.Diagnostics;

namespace MagicAR.Code
{
    internal class MarkRecognition : FrameLoop
    {
        public bool ShowDebugText = true;
        public bool UsePhoto = false;

        public int PlayerLifeOne = 20;
        public int PlayerLifeTwo = 20;

        private Matrix<float> intrisics;
        private Matrix<float> distCoeffs;
        private VideoCapture capture = new VideoCapture(1);


        List<List<List<byte>>> RotationShapes = new List<List<List<byte>>>();
        Matrix<float> Rotation = new Matrix<float>(3, 1);
        Matrix<float> RotationR = new Matrix<float>(3, 3);
        Matrix<float> Transform = new Matrix<float>(3, 1);
        Matrix<float> extrinsicMat = new Matrix<float>(4, 3);
        private MCvPoint3D32f[] obj;

        List<Byte> MarkerShape = new List<Byte>();  
        List<List<Byte>> ShapeCollection = new List<List<Byte>>();  
        public bool ShowOnce = true;

        Dictionary<ShapeType, List< List<byte>>> ShapeClasses = new Dictionary<ShapeType, List<List<byte>>>();

        public long SubtractionTimer;
        public long AdditionTimer;

        public Stopwatch stopWatch = new Stopwatch();
        Point PlayerOnePos;
        Point PlayerTwoPos;

        double DistanceTwoAdd;
        double DistanceOneAdd;

        double DistanceTwoSub;
        double DistanceOneSub;

        public MarkRecognition() {

            RotationShapes = GenerateShapes();
            UtilityAR.ReadIntrinsicsFromFile(out intrisics, out distCoeffs);

            stopWatch.Start();
        }

        public override void OnFrame()
        {
            if(AdditionTimer != null && SubtractionTimer != null)
            {
                Console.Clear();
                Console.WriteLine("Setup");
                Console.WriteLine(AdditionTimer.ToString());
                Console.WriteLine(SubtractionTimer.ToString());
            }

    

            if (DistanceOneAdd > DistanceTwoAdd)
            {
                EditLife(false, true);
            } else  {
                EditLife(true, true);

            }

            if (DistanceOneSub > DistanceTwoSub)
            {
                EditLife(false, false);
            }
            else {
                EditLife(true, false);
            }



            if (Console.KeyAvailable)
            {
                var key = Console.ReadKey(true);

                switch (key.Key)
                {
                    case ConsoleKey.W:
                        PlayerLifeOne++;
                        break;
                    case ConsoleKey.S:
                        PlayerLifeOne--;
                        break;


                    case ConsoleKey.E:
                        PlayerLifeTwo++;
                        break;
                    case ConsoleKey.D:
                        PlayerLifeTwo--;
                        break;
                } 
            }



            ShapeCollection.Clear();
            Mat frameOrigin = new Mat();
            Mat frameBinær = new Mat();
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat mapHiraki = new Mat();
            List<Mat> warpedFrame = new List<Mat>();
            VectorOfVectorOfPoint DestinationContours = new VectorOfVectorOfPoint();

            if (UsePhoto)
            {
                frameOrigin = CvInvoke.Imread("capture_1.jpg");
            } else
            {
                capture.Read(frameOrigin);
            }
           
            
            frameBinær = ConvertFrame(frameOrigin);

            VectorOfVectorOfPoint AcceptedContours = new VectorOfVectorOfPoint();
            VectorOfVectorOfPoint DistanationContours = new VectorOfVectorOfPoint();

            VectorOfPoint RefPositions = new VectorOfPoint();
            Point newPontA = new Point(0, 0);
            Point newPontB = new Point(300, 0);
            Point newPontC = new Point(300, 300);
            Point newPontD = new Point(0, 300);
            Point[] CompletedList = new Point[] { newPontA, newPontB, newPontC, newPontD };
            RefPositions.Push(CompletedList);

            List<Mat> Homography = new List<Mat>();
            CvInvoke.FindContours(frameBinær, contours, mapHiraki, RetrType.List, ChainApproxMethod.ChainApproxSimple);
            AcceptedContours = GenerateAcceptedContour(contours);

            for (var i = 0; i < AcceptedContours.Size; i++)
            {
                Homography.Add(CvInvoke.FindHomography(AcceptedContours[i], RefPositions, RobustEstimationAlgorithm.Ransac));
            }

            for (var i = 0; i < Homography.Count; i++)
            {
                Mat TempWarpedFrame = new Mat();
                CvInvoke.WarpPerspective(frameBinær, TempWarpedFrame, Homography[i], new Size(300, 300));
                warpedFrame.Add(TempWarpedFrame);
            }

            for (var i = 0; i < warpedFrame.Count; i++)
            {
                MarkerShape = new List<byte>();

                Byte[] ShapeData;
                int FramSize = (300 / 6);
                int HalfSize = FramSize / 2;

                for (var z = 0; z < 6; z++)
                {
                    for (var t = 0; t < 6; t++)
                    {

                        ShapeData = warpedFrame[i].GetRawData(new[] { (z * FramSize) + HalfSize, (t * FramSize) + HalfSize });
                        MarkerShape.Add(ShapeData[0]);
                    }
                }

                ShapeCollection.Add(MarkerShape);
 
            }


            for (var i = 0; i < ShapeCollection.Count; i++)
            {

                for (var m = 0; m < RotationShapes.Count; m++)
                {

                    for (var n = 0; n < 4; n++)
                    {
                        if (ShapeCollection[i].SequenceEqual(RotationShapes[m][n]))
                        {

                            MCvPoint3D32f newRotationA = new MCvPoint3D32f(0, 0, 0);
                            MCvPoint3D32f newRotationB = new MCvPoint3D32f(300, 0, 0);
                            MCvPoint3D32f newRotationC = new MCvPoint3D32f(300, 300, 0);
                            MCvPoint3D32f newRotationD = new MCvPoint3D32f(0, 300, 0);

                            switch (n)
                            {

                                case 1:
                                    newRotationA = new MCvPoint3D32f(0, 300, 0);
                                    newRotationB = new MCvPoint3D32f(0, 0, 0);
                                    newRotationC = new MCvPoint3D32f(300, 0, 0);
                                    newRotationD = new MCvPoint3D32f(300, 300, 0);
                                    break;

                                case 2:
                                    newRotationA = new MCvPoint3D32f(300, 300, 0);
                                    newRotationB = new MCvPoint3D32f(0, 300, 0);
                                    newRotationC = new MCvPoint3D32f(0, 0, 0);
                                    newRotationD = new MCvPoint3D32f(300, 0, 0);
                                    break;

                                case 3:
                                    newRotationA = new MCvPoint3D32f(300, 0, 0);
                                    newRotationB = new MCvPoint3D32f(300, 300, 0);
                                    newRotationC = new MCvPoint3D32f(0, 300, 0);
                                    newRotationD = new MCvPoint3D32f(0, 0, 0);

                                    break;
                            }

                            //Verdens punkterne,
                            obj = new MCvPoint3D32f[] { newRotationA, newRotationB, newRotationC, newRotationD };


                            Point[] convertedContours = AcceptedContours[i].ToArray();
                            PointF[] newPoints = new PointF[] { new PointF(0, 0), new PointF(0, 0), new PointF(0, 0), new PointF(0, 0) };

                            for (var q = 0; q < convertedContours.Length; q++)
                            {
                                PointF TempPoint = convertedContours[q];
                                newPoints[q] = TempPoint;

                            }


                            CvInvoke.SolvePnP(obj, newPoints, intrisics, distCoeffs, Rotation, Transform);
                            CvInvoke.Rodrigues(Rotation, RotationR);
                            float[,] rValues = RotationR.Data;
                            float[,] tValues = Transform.Data;
                            extrinsicMat = new Matrix<float>(new float[,]
                            {
                                {rValues[0,0],rValues[0,1],rValues[0,2],tValues[0,0]},
                                {rValues[1,0],rValues[1,1],rValues[1,2],tValues[1,0]},
                                {rValues[2,0],rValues[2,1],rValues[2,2],tValues[2,0]},
                            });


            
                            Matrix<float> textPoint = new Matrix<float>(new float[4] { 0, 150, 0, 1 });

                            Point textPointStart = UtilityAR.WorldToScreen(textPoint, intrisics * extrinsicMat);
                           

                            
                            
                            MCvScalar color = new MCvScalar(100, 0, 100);

                         
                            
                            if( Rotation[2, 0] < -0.15 && Rotation[2, 0] > -2.6)
                            {  color = new MCvScalar(0, 100, 100);                                    
                            } else  if(Rotation[2, 0] > 0.15 && Rotation[2, 0] < 2.6)
                            { color = new MCvScalar(0, 100, 100);
                            }





                            List<List<byte>> a = ShapeClasses[ShapeType.LcOne];
                            List<byte> b = ShapeCollection[i];

                            if(ShapeClasses[ShapeType.LcOne].Any(a => a.SequenceEqual(ShapeCollection[i]))){
                                color = new MCvScalar(0, 255, 0);
                                CvInvoke.PutText(frameOrigin, PlayerLifeOne.ToString(), textPointStart, FontFace.HersheyTriplex, 2, color);
                                PlayerOnePos = textPointStart;

                              

                            }

                            if (ShapeClasses[ShapeType.LcTwo].Any(a => a.SequenceEqual(ShapeCollection[i])))
                            {
                                color = new MCvScalar(0, 0, 255);
                                CvInvoke.PutText(frameOrigin, PlayerLifeTwo.ToString(), textPointStart, FontFace.HersheyTriplex, 2, color);
                                PlayerTwoPos = textPointStart;
                            }

                            if (ShapeClasses[ShapeType.Ceature].Any(a => a.SequenceEqual(ShapeCollection[i])))
                            {
                                //CvInvoke.PutText(frameOrigin, "Creature", textPointStart, FontFace.HersheyTriplex, 2, color);
                                GameShapes.DrawCube(frameOrigin, intrisics * extrinsicMat, color, 300);
                            }

                            if (ShapeClasses[ShapeType.Land].Any(a => a.SequenceEqual(ShapeCollection[i])))
                            {
                                //CvInvoke.PutText(frameOrigin, "Land", textPointStart, FontFace.HersheyTriplex, 2, color);
                                GameShapes.DrawTriangle(frameOrigin, intrisics * extrinsicMat, color, 300);
                            }

                            if (ShapeClasses[ShapeType.AddLife].Any(a => a.SequenceEqual(ShapeCollection[i])))
                            {
                                color = new MCvScalar(120, 230, 230);
                                CvInvoke.PutText(frameOrigin, "+", textPointStart, FontFace.HersheyTriplex, 3, color);
                                AdditionTimer = stopWatch.ElapsedMilliseconds;

                                DistanceOneAdd = Math.Round(Math.Sqrt(Math.Pow((textPointStart.X - PlayerOnePos.X), 2) + Math.Pow((textPointStart.Y - PlayerOnePos.Y), 2)), 1);
                                DistanceTwoAdd = Math.Round(Math.Sqrt(Math.Pow((textPointStart.X - PlayerTwoPos.X), 2) + Math.Pow((textPointStart.Y - PlayerTwoPos.Y), 2)), 1);




                            }

                            if (ShapeClasses[ShapeType.SubLife].Any(a => a.SequenceEqual(ShapeCollection[i])))
                             {
                                 color = new MCvScalar(120, 230, 230);
                                 CvInvoke.PutText(frameOrigin, "-", textPointStart, FontFace.HersheyTriplex, 3, color);
                                 SubtractionTimer = stopWatch.ElapsedMilliseconds;

                                DistanceOneSub = Math.Round(Math.Sqrt(Math.Pow((textPointStart.X - PlayerOnePos.X), 2) + Math.Pow((textPointStart.Y - PlayerOnePos.Y), 2)), 1);
                                DistanceTwoSub = Math.Round(Math.Sqrt(Math.Pow((textPointStart.X - PlayerTwoPos.X), 2) + Math.Pow((textPointStart.Y - PlayerTwoPos.Y), 2)), 1);


                          
                            }



                        }
                    }
                }
            }

            CvInvoke.DrawContours(frameOrigin, AcceptedContours, -1, new MCvScalar(0, 255, 100));
            CvInvoke.Imshow("MarkAR Window", frameOrigin);

        }

        public void EditLife(bool isPlayerOne,bool isAdd)
        {
            if (isPlayerOne)
            {
                if (isAdd)
                {
                    if (stopWatch.ElapsedMilliseconds - AdditionTimer > 1000)
                    {
                        PlayerLifeOne += 1;
                        AdditionTimer = stopWatch.ElapsedMilliseconds - 500;
                    }

                } else
                {
                    if (stopWatch.ElapsedMilliseconds - SubtractionTimer > 1000)
                    {
                        PlayerLifeOne -= 1;
                        SubtractionTimer = stopWatch.ElapsedMilliseconds - 500;
                    }
                }

            
            } else
            {
                if (isAdd)
                {
                    if (stopWatch.ElapsedMilliseconds - AdditionTimer > 1000)
                    {
                        PlayerLifeTwo += 1;
                        AdditionTimer = stopWatch.ElapsedMilliseconds - 500;
                    }
                } else
                {
                    if (stopWatch.ElapsedMilliseconds - SubtractionTimer > 1000)
                    {
                        PlayerLifeTwo -= 1;
                        SubtractionTimer = stopWatch.ElapsedMilliseconds - 500;
                    }
                }

             
            }
     

            
        }

        public Mat ConvertFrame(Mat OriginalFrame)
        {
            Mat frameGray = new Mat(); // << GraySclae Frame
            Mat result = new Mat(); // Binær Frame
            CvInvoke.CvtColor(OriginalFrame, frameGray, ColorConversion.Bgr2Gray);
            CvInvoke.Threshold(frameGray, result, 0, 255, ThresholdType.Otsu);

            return result;

        }

        public VectorOfVectorOfPoint GenerateAcceptedContour(VectorOfVectorOfPoint contours)
        {
            VectorOfVectorOfPoint Result = new VectorOfVectorOfPoint();

            for (int i = 0; i < contours.Size; i++)
            {
                VectorOfPoint contour = contours[i];
                VectorOfPoint approxContour = new VectorOfPoint();

                CvInvoke.ApproxPolyDP(contour, approxContour, 4, true);

                if (approxContour.Size == 4)
                {
                    Result.Push(approxContour);
                }
            }

            return Result;
        }

        public List<List<List<byte>>> GenerateShapes()
        {

            List<List<Matrix<byte>>> ShapeCollection = new List<List<Matrix<byte>>>();

            Matrix<Byte> Type1 = new Matrix<byte>(new byte[,] {
                                   {0,0,0,0,0,0},
                                   {0,0,0,255,255,0},
                                   {0,255,0,255,255,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,255,255,0},
                                   {0,0,0,0,0,0}});

            Matrix<Byte> Type2 = new Matrix<byte>(new byte[,] {
                                   {0,0,0,0,0,0},
                                   {0,255,255,0,0,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,255,255,0},
                                   {0,0,0,0,0,0}});

            Matrix<Byte> Type3 = new Matrix<byte>(new byte[,] {
                                   {0,0,0,0,0,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,0,0,0},
                                   {0,0,0,0,0,0}});

            Matrix<Byte> Type4 = new Matrix<byte>(new byte[,] {
                                   {0,0,0,0,0,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,255,255,0},
                                   {0,255,0,255,255,0},
                                   {0,255,0,255,255,0},
                                   {0,0,0,0,0,0}});

            Matrix<Byte> Type5 = new Matrix<byte>(new byte[,] {
                                   {0,0,0,0,0,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,0,0,0},
                                   {0,255,255,255,0,0},
                                   {0,0,0,0,0,0}});

            Matrix<Byte> Type6 = new Matrix<byte>(new byte[,] {
                                   {0,0,0,0,0,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,0,0,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,255,255,0},
                                   {0,0,0,0,0,0}});

            Matrix<Byte> Type7 = new Matrix<byte>(new byte[,] {
                                   {0,0,0,0,0,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,255,0,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,255,255,0},
                                   {0,0,0,0,0,0}});

            Matrix<Byte> Type8 = new Matrix<byte>(new byte[,] {
                                   {0,0,0,0,0,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,255,255,0},
                                   {0,255,255,255,255,0},
                                   {0,255,0,255,255,0},
                                   {0,0,0,0,0,0}});


            ShapeCollection.Add(RotateShapes(Type1));
            ShapeCollection.Add(RotateShapes(Type2));
            ShapeCollection.Add(RotateShapes(Type3));
            ShapeCollection.Add(RotateShapes(Type4));
            ShapeCollection.Add(RotateShapes(Type5));
            ShapeCollection.Add(RotateShapes(Type6));



            List<List<List<byte>>> ConShapeCollection = new List<List<List<byte>>>();

            for (var i = 0; i < ShapeCollection.Count; i++)
            {
                List<List<byte>> TempList = new List<List<byte>>();

                for (var m = 0; m < ShapeCollection[i].Count; m++)
                {
                    List<byte> TempByteList = new List<byte>();

                    for (var z = 0; z < 6; z++)
                    {
                        for (var y = 0; y < 6; y++)
                        {
                            TempByteList.Add(ShapeCollection[i][m][z, y]);
                        }
                    }
                    TempList.Add(TempByteList);
                }

                ConShapeCollection.Add(TempList);
            }

            ShapeClasses.Add(ShapeType.LcOne, ConShapeCollection[0]);
            ShapeClasses.Add(ShapeType.LcTwo, ConShapeCollection[1]);
            ShapeClasses.Add(ShapeType.Ceature, ConShapeCollection[2]);
            ShapeClasses.Add(ShapeType.Land, ConShapeCollection[3]);
            ShapeClasses.Add(ShapeType.AddLife, ConShapeCollection[4]);
            ShapeClasses.Add(ShapeType.SubLife, ConShapeCollection[5]);
            return ConShapeCollection;

        }
        public List<Matrix<Byte>> RotateShapes(Matrix<Byte> BaseShape)
        {
            List<Matrix<Byte>> Shape = new List<Matrix<Byte>>();

            Matrix<byte> Result90 = new Matrix<byte>(new byte[6, 6]);
            Matrix<byte> Result180 = new Matrix<byte>(new byte[6, 6]);
            Matrix<byte> Result270 = new Matrix<byte>(new byte[6, 6]);




            Shape.Add(BaseShape);

            CvInvoke.Rotate(BaseShape, Result90, RotateFlags.Rotate90Clockwise);
            Shape.Add(Result90);
            CvInvoke.Rotate(Result90, Result180, RotateFlags.Rotate90Clockwise);
            Shape.Add(Result180);
            CvInvoke.Rotate(Result180, Result270, RotateFlags.Rotate90Clockwise);
            Shape.Add(Result270);

            return Shape;
        }
    }
}

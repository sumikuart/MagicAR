using MagicAR.Code;
using System.Drawing;

internal class Program
{
    static void Main(string[] args)
    {
        Size gridSize = new Size(7, 4);

        //UtilityAR.CaptureLoop(gridSize,1); //< Tager billeder
        //UtilityAR.CalibrateCamera(gridSize); //< Calibere fra bilelderne

        //CheesAR chreesRender = new CheesAR();
        //chreesRender.Run();

        MarkRecognition markRender = new MarkRecognition();
        markRender.Run();

    }
}
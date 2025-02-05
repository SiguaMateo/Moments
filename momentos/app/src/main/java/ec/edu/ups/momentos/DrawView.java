package ec.edu.ups.momentos;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

public class DrawView extends View {
    private Paint paint;
    private Path path;
    private Bitmap bitmap;
    private Canvas bitmapCanvas;

    public DrawView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        paint = new Paint();
        paint.setColor(0xFF000000); // Color negro
        paint.setStyle(Paint.Style.STROKE); // Sólo contornos
        paint.setStrokeWidth(10f); // Ancho del pincel

        path = new Path();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (bitmap == null) {
            // Inicializa el Bitmap en formato RGB_565 (3 canales)
            bitmap = Bitmap.createBitmap(getWidth(), getHeight(), Bitmap.Config.RGB_565);
            bitmapCanvas = new Canvas(bitmap);
            bitmap.eraseColor(0xFFFFFFFF); // Fondo blanco inicial
        }
        bitmapCanvas.drawPath(path, paint); // Dibuja en el Bitmap
        canvas.drawBitmap(bitmap, 0, 0, null); // Renderiza el Bitmap en la vista
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                path.moveTo(x, y); // Inicia el dibujo
                break;
            case MotionEvent.ACTION_MOVE:
                path.lineTo(x, y); // Dibuja líneas a medida que se mueve el dedo
                break;
        }
        invalidate(); // Redibuja la vista
        return true;
    }

    /**
     * Devuelve el Bitmap generado.
     * @return Un Bitmap en formato RGB_565.
     */
    public Bitmap getBitmap() {
        return bitmap;
    }

    /**
     * Limpia el lienzo y reinicia el Path.
     */
    public void clearCanvas() {
        path.reset();
        if (bitmap != null) {
            bitmap.eraseColor(0xFFFFFFFF); // Limpia el Bitmap con color blanco
        }
        invalidate();
    }
}
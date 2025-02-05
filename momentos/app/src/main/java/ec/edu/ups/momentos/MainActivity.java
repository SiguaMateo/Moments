package ec.edu.ups.momentos;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    // Carga la librería nativa
    static {
        System.loadLibrary("native-lib");
    }

    private DrawView drawView;
    private TextView textView;

    // Firma del método nativo: recibe el Bitmap del dibujo y el AssetManager para acceder a los assets
    private native String procesarDibujo(Bitmap bitmap, AssetManager assetManager);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        drawView = findViewById(R.id.drawView);
        textView = findViewById(R.id.textView);
        Button btnClasificar = findViewById(R.id.btnClasificar);
        Button btnLimpiar = findViewById(R.id.btnLimpiar);

        btnClasificar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap bitmap = drawView.getBitmap();
                if (bitmap != null) {
                    // Se llama al método nativo pasando el Bitmap y el AssetManager
                    String classification = procesarDibujo(bitmap, getAssets());
                    textView.setText("Clasificación: " + classification);
                } else {
                    textView.setText("Error: No se pudo obtener el dibujo.");
                }
            }
        });

        btnLimpiar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                drawView.clearCanvas();
                textView.setText("Dibuja una figura");
            }
        });
    }
}

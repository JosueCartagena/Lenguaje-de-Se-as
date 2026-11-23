import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "bo.lsb.aimara"
    compileSdk = 36

    defaultConfig {
        applicationId = "bo.lsb.aimara"
        minSdk = 28
        // Solo procesadores de celulares reales (ARM): el APK pesa la mitad.
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    // Los modelos se leen directo del APK: no comprimirlos
    androidResources {
        noCompress += listOf("tflite", "task")
    }
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_11) }
}

// Descarga automática del modelo de MediaPipe (holistic_landmarker.task) al compilar por primera vez
extra["ASSET_DIR"] = "$projectDir/src/main/assets"
apply(from = "download_tasks.gradle.kts")

dependencies {
    implementation("androidx.core:core-ktx:1.18.0")
    implementation("androidx.activity:activity-ktx:1.12.4")

    // Cámara
    implementation("androidx.camera:camera-core:1.5.3")
    implementation("androidx.camera:camera-camera2:1.5.3")
    implementation("androidx.camera:camera-lifecycle:1.5.3")
    implementation("androidx.camera:camera-view:1.5.3")

    // Puntos del cuerpo, manos y cara (MediaPipe Holistic Landmarker)
    implementation("com.google.mediapipe:tasks-vision:1.2.2")

    // Nuestra red neuronal (TensorFlow Lite / LiteRT)
    implementation("com.google.ai.edge.litert:litert:1.0.1")
}

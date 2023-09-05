import subprocess

def ejecutar_identificador_movimiento():
    subprocess.call(["python", "E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\Proyecto1\\IdentificadorMovimiento.py"])

def ejecutar_sin_movimiento():
    subprocess.call(["python", "E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\Proyecto1\\ProyectoMediaPipe_SinMovimiento.py"])

def main():
    print("Seleccione una opción:")
    print("1. Reconocimiento de letras con movimiento")
    print("2. Reconocimiento de letras sin movimiento")

    opcion = input("Ingrese el número de opción: ")

    if opcion == "1":
        ejecutar_identificador_movimiento()
    elif opcion == "2":
        ejecutar_sin_movimiento()
    else:
        print("Opción inválida. Por favor, seleccione 1 o 2.")

if __name__ == "__main__":
    main()
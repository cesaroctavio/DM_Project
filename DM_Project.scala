
//Imports de scala
//Importar un sesion de Spark
import org.apache.spark.sql.SparkSession
//Import de codigo para reportar error reducidos
import org.apache.log4j._

//Import de funciones de sql
import org.apache.spark.sql.functions._

//Import de Vector Assembler and Vector
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

//Impor de la libreria de Kmeans para el algoritmo de agrupamiento
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.BisectingKMeans

//Inicio de sesion en spark
val spark = SparkSession.builder().getOrCreate()

//Codigo para reportar error reducidos
Logger.getLogger("org").setLevel(Level.ERROR)

//Cargar el dataset
val df = spark.read.option("inferSchema","true").csv("Iris.csv")
df.printSchema()
df.show()

//Limpieza de datos
//Funciones definidas por el usuario (también conocido como UDF ) es una característica de Spark SQL para definir nuevas funciones basadas en columnas que amplían el vocabulario de DSL de Spark SQL para transformar conjuntos de datos .
val Convert = udf[Double, String](_.toDouble)

val df2 = df.
withColumn("_c0",Convert(df("_c0"))).
withColumn("_c1",Convert(df("_c1"))).
withColumn("_c2",Convert(df("_c2"))).
withColumn("_c3",Convert(df("_c3"))).
select($"_c0".as("SepalLength"),$"_c1".as("SepalWidth"),$"_c2".as("PetalLength"),$"_c3".as("PetalWidth"))

df2.show()

//Se crea un nuevo objeto Vector Assembeler para las columnas de caracteristicas como un conjunto de entrada
val assembler = new VectorAssembler().setInputCols(Array("SepalLength","SepalWidth","PetalLength","PetalWidth")).setOutputCol("features")

//Objeto assembler para transformar
val outer_data = assembler.transform(df2).select($"features")
outer_data.show()

//Aplicando K-Means
val kmeans = new KMeans().setK(5).setSeed(1L)
val model = kmeans.fit(outer_data)

//Evaluar los grupos usnado WSSE y mostrar el resultado
val WSSE = model.computeCost(outer_data)
println(s"Within set sum of Squared Errors = $WSSE")

//Mostrar resultados de los Cluster
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

//Aplicando Bisecting K-KMeans
val bkm = new BisectingKMeans().setK(5).setSeed(1L)
val bmodel = bkm.fit(outer_data)

//Evaluar los grupos usnado WSSE y mostrar el resultado
val bkm_WSSE = bmodel.computeCost(outer_data)
println(s"Within Set Sum of Squared Errors = $bkm_WSSE")

//Mostrar resultados de los Cluster
println("Cluster Centers: ")
val centers = bmodel.clusterCenters
centers.foreach(println)


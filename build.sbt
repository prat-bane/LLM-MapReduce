ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.3.3"

lazy val root = (project in file("."))
  .settings(
    name := "ScalaRest"
  )

fork := true

javaOptions += "-Djavacpp.platform=windows-x86_64"

libraryDependencies ++= Seq(
  "org.tensorflow" % "tensorflow-core-platform" % "0.4.2",
  "com.knuddels" % "jtokkit" % "1.1.0",
  "org.tensorflow" % "ndarray" % "1.0.0",
  "org.apache.hadoop" % "hadoop-common" % "3.3.4",
  "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.3.4",
  "org.apache.hadoop" % "hadoop-mapreduce-client-jobclient" % "3.3.4",
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta7",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta7",
  "org.datavec" % "datavec-api" % "1.0.0-beta7",
  "log4j" % "log4j" % "1.2.17"
)

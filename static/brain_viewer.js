
import * as THREE from "https://cdn.skypack.dev/three";
import { GLTFLoader } from "https://cdn.skypack.dev/three/examples/jsm/loaders/GLTFLoader.js";

const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(75,innerWidth/innerHeight,0.1,1000);
const renderer=new THREE.WebGLRenderer();
renderer.setSize(innerWidth,innerHeight);
document.body.appendChild(renderer.domElement);

camera.position.z=4;

const loader=new GLTFLoader();

loader.load("/static/brain_mri.glb",function(gltf){
 scene.add(gltf.scene);
});

function animate(){
 requestAnimationFrame(animate);
 renderer.render(scene,camera);
}

animate();

document.getElementById("upload").addEventListener("change",async function(){

 const file=this.files[0];
 const fd=new FormData();
 fd.append("image",file);

 const res=await fetch("/analyze",{method:"POST",body:fd});
 const data=await res.json();

 console.log("Brain activations:",data.activations);

});

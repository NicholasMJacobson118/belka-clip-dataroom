// ─── Beating TEE VALVE volume render (CT–TEE fusion) ────────────────────────
// Ray-marches the masked-TEE valve sequence (window.VALVE_VOLSEQ, 12 frames, echo
// amplitude, already in CT/scene space) with an ULTRASOUND transfer function, fused
// over the CT heart volume. Mirrors heart_volseq_render.js. Exposes VALVE_VOLSEQ_R
// and a "4D Valve (TEE)" toggle that swaps the static surface leaflets for the
// beating echo volume. Depends on globals: THREE, renderer, scene, camera, HM.
(function () {
  'use strict';
  var SEQ = window.VALVE_VOLSEQ;
  if (!SEQ) { console.warn('VALVE_VOLSEQ missing.'); return; }
  if (!(renderer && renderer.capabilities && renderer.capabilities.isWebGL2)) {
    console.warn('WebGL2 unavailable — valve volume disabled.'); return;
  }
  var W = SEQ.dims[0], H = SEQ.dims[1], D = SEQ.dims[2];
  var Tex3D = THREE.Data3DTexture || THREE.DataTexture3D;

  // EXACT Slicer TEE volume-rendering preset (from slicer_fusion_vr.py set_tee_transfer_functions),
  // defined over echo amplitude 0..255. Our baked texture value d maps back to amplitude as
  // amp = 8 + d*247 (the build window), so LUT[n] is sampled at that amplitude.
  var OPAC = [[0,0.00],[15,0.00],[40,0.10],[120,0.20],[180,0.40],[220,0.65],[255,0.80]];
  var COL  = [[0,0,0,0],[30,0.2,0.1,0.0],[80,0.6,0.4,0.1],[150,0.9,0.7,0.2],[220,1.0,0.9,0.5],[255,1.0,1.0,0.8]];
  function interp1(p,a){if(a<=p[0][0])return p[0][1];for(var i=1;i<p.length;i++)if(a<=p[i][0]){var x=p[i-1],y=p[i],f=(a-x[0])/(y[0]-x[0]);return x[1]+(y[1]-x[1])*f;}return p[p.length-1][1];}
  function interp3(p,a){if(a<=p[0][0])return[p[0][1],p[0][2],p[0][3]];for(var i=1;i<p.length;i++)if(a<=p[i][0]){var x=p[i-1],y=p[i],f=(a-x[0])/(y[0]-x[0]);return[x[1]+(y[1]-x[1])*f,x[2]+(y[2]-x[2])*f,x[3]+(y[3]-x[3])*f];}var l=p[p.length-1];return[l[1],l[2],l[3]];}
  var LN = 256, lut = new Uint8Array(LN*4);
  for (var n=0;n<LN;n++){var a=8+(n/(LN-1))*247;var c=interp3(COL,a),o=interp1(OPAC,a);
    lut[n*4]=Math.round(c[0]*255);lut[n*4+1]=Math.round(c[1]*255);lut[n*4+2]=Math.round(c[2]*255);lut[n*4+3]=Math.round(o*255);}
  var lutTex=new THREE.DataTexture(lut,LN,1,THREE.RGBAFormat);
  lutTex.minFilter=lutTex.magFilter=THREE.LinearFilter; lutTex.needsUpdate=true;

  function b64ToU8(b64){var bin=atob(b64),u=new Uint8Array(bin.length);for(var i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
  function gunzip(b64){var u8=b64ToU8(b64);if(!SEQ.gz)return Promise.resolve(u8);
    var s=new Blob([u8]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Response(s).arrayBuffer().then(function(ab){return new Uint8Array(ab);});}
  function makeVolTex(data){var t=new Tex3D(data,W,H,D);t.format=THREE.RedFormat;t.type=THREE.UnsignedByteType;
    t.minFilter=THREE.LinearFilter;t.magFilter=THREE.LinearFilter;t.wrapR=t.wrapS=t.wrapT=THREE.ClampToEdgeWrapping;t.unpackAlignment=1;t.needsUpdate=true;return t;}

  var mn=SEQ.sceneMin, sz=SEQ.sceneSize;
  var uniforms={
    uVol:{value:null}, uLUT:{value:lutTex},
    uBoxMin:{value:new THREE.Vector3(mn[0],mn[1],mn[2])}, uBoxSize:{value:new THREE.Vector3(sz[0],sz[1],sz[2])},
    uDensity:{value:1.0}, uShade:{value:0.85}, uSteps:{value:192},
    uLightDir:{value:new THREE.Vector3(0.4,-0.7,0.7).normalize()}
  };
  var vert='out vec3 vWorld;\nvoid main(){vec4 wp=modelMatrix*vec4(position,1.0);vWorld=wp.xyz;gl_Position=projectionMatrix*viewMatrix*wp;}';
  var frag=[
    'precision highp float; precision highp sampler3D;',
    'uniform sampler3D uVol; uniform sampler2D uLUT;',
    'uniform vec3 uBoxMin,uBoxSize,uLightDir; uniform float uDensity,uShade; uniform int uSteps;',
    'in vec3 vWorld; out vec4 outColor;',
    'void main(){',
    '  vec3 ro=cameraPosition; vec3 rd=normalize(vWorld-ro);',
    '  vec3 bmin=uBoxMin,bmax=uBoxMin+uBoxSize;',
    '  vec3 t0=(bmin-ro)/rd,t1=(bmax-ro)/rd; vec3 tn=min(t0,t1),tf=max(t0,t1);',
    '  float tnear=max(max(max(tn.x,tn.y),tn.z),0.0); float tfar=min(min(tf.x,tf.y),tf.z);',
    '  if(tnear>=tfar) discard;',
    '  int N=uSteps; float dt=(tfar-tnear)/float(N);',
    '  vec3 eps=1.0/vec3(textureSize(uVol,0));',
    '  vec3 acc=vec3(0.0); float aAcc=0.0;',
    '  float jit=fract(sin(dot(gl_FragCoord.xy,vec2(12.9898,78.233)))*43758.5453);',
    '  float tcur=tnear+dt*(0.5+0.5*jit);',
    '  vec3 L=normalize(uLightDir);',
    '  for(int i=0;i<512;i++){ if(i>=N) break;',
    '    vec3 p=ro+rd*tcur; vec3 tc=(p-uBoxMin)/uBoxSize;',
    '    float d=texture(uVol,tc).r;',
    '    if(d>0.02){',
    '      vec4 tf=texture(uLUT, vec2(clamp(d,0.0,1.0),0.5));',
    '      float a=tf.a*uDensity;',
    '      if(a>0.002){',
    '        float gx=texture(uVol,tc+vec3(eps.x,0,0)).r-texture(uVol,tc-vec3(eps.x,0,0)).r;',
    '        float gy=texture(uVol,tc+vec3(0,eps.y,0)).r-texture(uVol,tc-vec3(0,eps.y,0)).r;',
    '        float gz=texture(uVol,tc+vec3(0,0,eps.z)).r-texture(uVol,tc-vec3(0,0,eps.z)).r;',
    '        vec3 g=vec3(gx,gy,gz); float gl=length(g);',
    '        vec3 nrm=gl>0.0002?-g/gl:L;',
    '        float diff=max(dot(nrm,L),0.0);',
    '        vec3 Vd=-rd; vec3 Hh=normalize(L+Vd); float spec=pow(max(dot(nrm,Hh),0.0),10.0)*0.2;',
    '        float lit=0.1+0.8*diff;',
    '        vec3 col=tf.rgb*mix(1.0,lit,uShade)+vec3(spec)*uShade;',
    '        col*=a; acc+=(1.0-aAcc)*col; aAcc+=(1.0-aAcc)*a;',
    '        if(aAcc>0.98) break;',
    '      }',
    '    }',
    '    tcur+=dt;',
    '  }',
    '  if(aAcc<0.003) discard;',
    '  outColor=vec4(acc,aAcc);',
    '}'
  ].join('\n');

  var mat=new THREE.ShaderMaterial({uniforms:uniforms,vertexShader:vert,fragmentShader:frag,
    glslVersion:THREE.GLSL3,side:THREE.BackSide,transparent:true,depthTest:false,depthWrite:false});
  var mesh=new THREE.Mesh(new THREE.BoxGeometry(sz[0],sz[1],sz[2]),mat);
  mesh.position.set(mn[0]+sz[0]/2,mn[1]+sz[1]/2,mn[2]+sz[2]/2);
  mesh.renderOrder=-9; mesh.frustumCulled=false; mesh.visible=false;   // over the heart (-10)
  scene.add(mesh);

  var texs=new Array(SEQ.nFrames).fill(null), ready=false, cur=-1;
  Promise.all(SEQ.frames.map(function(b64,i){return gunzip(b64).then(function(u8){texs[i]=makeVolTex(u8);});}))
    .then(function(){ready=true;uniforms.uVol.value=texs[0];cur=0;});

  var beat={play:false,cycleMs:1500,t0:performance.now()};
  function tick(now){requestAnimationFrame(tick);if(!ready||!beat.play)return;
    // phase-lock to the heart's cardiac clock so CT and TEE move in the same time
    var ph=(window.HEART_VOLSEQ_R&&HEART_VOLSEQ_R.phase)?HEART_VOLSEQ_R.phase():(((now-beat.t0)%beat.cycleMs)/beat.cycleMs);
    var f=Math.floor(ph*SEQ.nFrames)%SEQ.nFrames;
    if(f!==cur){cur=f;uniforms.uVol.value=texs[f];}}
  requestAnimationFrame(tick);

  // Hide/show the static surface leaflets so we don't see a double valve.
  function setSurfaceHidden(hide){
    if(!window.HM) return;
    ['aml','pml','annulus'].forEach(function(k){var mat2=HM[k];if(!mat2)return;
      scene.traverse(function(o){if(o.isMesh&&o.material===mat2)o.visible=!hide;});});
  }
  var on=false;
  function setPlaying(p){
    on=p; mesh.visible=p&&ready; beat.play=p; if(p)beat.t0=performance.now();
    setSurfaceHidden(p);
    if(btn){btn.style.opacity=p?'0.75':'1';btn.style.background=p?'#241a3a':'#1a1c2e';}
  }

  var btn=document.createElement('button');
  btn.id='btn-4dvalve'; btn.textContent='♡ 4D Valve (TEE volume)';
  btn.title='Show the real TEE valve as a beating echo volume, fused in the CT heart';
  btn.style.cssText='position:absolute;top:108px;right:12px;padding:5px 12px;'+
    'background:#1a1c2e;border:1px solid #8a5ad0;color:#b89af0;border-radius:4px;'+
    'cursor:pointer;font-size:11px;z-index:20;letter-spacing:0.04em;';
  btn.onclick=function(){setPlaying(!on);};
  document.getElementById('viewport').appendChild(btn);

  window.VALVE_VOLSEQ_R={
    mesh:mesh, uniforms:uniforms,
    setVisible:function(v){mesh.visible=v&&ready;},
    setPlaying:setPlaying, isPlaying:function(){return on;},
    setOpacity:function(v){uniforms.uDensity.value=v;},
    setRate:function(ms){beat.cycleMs=ms;}
  };

  // tuning slider (no rebuild needed): valve echo brightness/opacity
  (function(){var card=document.getElementById('heart-ctrl-card');if(!card)return;
    var w=document.createElement('div');w.className='slider-wrap';
    var lr=document.createElement('div');lr.className='slider-label';
    var lv=document.createElement('span');lv.className='slider-val';lv.textContent='100%';
    lr.appendChild(document.createTextNode('Valve echo '));lr.appendChild(lv);
    var inp=document.createElement('input');inp.type='range';inp.min=30;inp.max=250;inp.step=1;inp.value=100;
    inp.oninput=function(){var v=parseFloat(inp.value);lv.textContent=Math.round(v)+'%';uniforms.uDensity.value=v/100;};
    w.appendChild(lr);w.appendChild(inp);card.appendChild(w);})();
})();

// Pricing explorer — mirrors src/pricing.py (Normal CDF, push band, quarter-line, de-vig, EV).
function erf(x){var s=x<0?-1:1;x=Math.abs(x);var t=1/(1+0.3275911*x);
var y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);
return s*y;}
function cdf(x,m,sd){return 0.5*(1+erf((x-m)/(sd*Math.SQRT2)));}
function frac(x){return x-Math.floor(x);}
function overUnder(mu,sd,line){sd=Math.max(sd,1e-6);var f=Math.round(frac(line)*10000)/10000;
 if(f===0.25||f===0.75){var a=overUnder(mu,sd,line-0.25),b=overUnder(mu,sd,line+0.25);
   return[(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2];}
 if(Math.abs(f-0.5)<1e-6){var pu=cdf(line,mu,sd);return[1-pu,pu,0];}
 var lo=cdf(line-0.5,mu,sd),hi=cdf(line+0.5,mu,sd);return[1-hi,lo,hi-lo];}
function devig(o,u){if(!o||!u)return[null,null];var io=1/o,iu=1/u,s=io+iu;return[io/s,iu/s];}
function fmtOdds(p){return p>1e-9?(1/p).toFixed(2):'–';}

var MODEL=null;
fetch('data/model.json').then(r=>r.json()).then(d=>{MODEL=d;initLab();}).catch(()=>{});

function sigmaFor(t,mu){var d=MODEL.dispersion[t];return Math.max(d.alpha+d.beta*mu,d.sigma_floor);}

function initLab(){
 var sel=document.getElementById('lab-stat');if(!sel)return;
 var ts=Object.keys(MODEL.dispersion);
 ts.forEach(function(t){var o=document.createElement('option');o.value=t;
   o.textContent=MODEL.target_label[t]||t;sel.appendChild(o);});
 sel.value=ts.indexOf('tackles')>=0?'tackles':ts[0];
 ['lab-stat','lab-mean','lab-line','lab-over','lab-under'].forEach(function(id){
   document.getElementById(id).addEventListener('input',function(e){if(id==='lab-stat')presetFor(sel.value);render();});});
 presetFor(sel.value);render();
}
function presetFor(t){
 var mean=MODEL.typical_mean[t]||10, max=Math.max(8,Math.ceil(mean*2.2));
 var mu=document.getElementById('lab-mean'),li=document.getElementById('lab-line');
 mu.max=max;li.max=max;mu.value=mean;li.value=Math.max(0,Math.round((mean-1.5)*2)/2);
}
function render(){
 var t=document.getElementById('lab-stat').value;
 var mu=parseFloat(document.getElementById('lab-mean').value);
 var line=parseFloat(document.getElementById('lab-line').value);
 var over=parseFloat(document.getElementById('lab-over').value)||null;
 var under=parseFloat(document.getElementById('lab-under').value)||null;
 var sd=sigmaFor(t,mu);
 document.getElementById('lab-mean-v').textContent=mu.toFixed(1);
 document.getElementById('lab-line-v').textContent=line.toFixed(1);
 var pr=overUnder(mu,sd,line),pOver=pr[0],pUnder=pr[1],push=pr[2];
 var mk=devig(over,under),mOver=mk[0];
 var evOver=over?(pOver*over+push-1):null, evUnder=under?(pUnder*under+push-1):null;
 var best=(evOver||-9)>=(evUnder||-9)?['OVER',evOver,pOver,over]:['UNDER',evUnder,pUnder,under];
 drawCurve(mu,sd,line);
 var out=document.getElementById('lab-out');
 function stat(cls,v,l){return '<div class="ostat '+cls+'"><b>'+v+'</b><span>'+l+'</span></div>';}
 var edge=(mOver!=null)?((pOver-mOver)*100):null;
 var verdict;
 if(best[1]==null){verdict='<div class="verdict lose">Enter a book price to see edge & EV.</div>';}
 else if(best[1]>0.001){verdict='<div class="verdict win">Model sees value on the '+best[0]+
   ' @ '+best[3].toFixed(2)+' — '+(best[1]*100).toFixed(1)+'% EV.</div>';}
 else{verdict='<div class="verdict lose">No edge at these prices ('+best[0]+' EV '+
   (best[1]*100).toFixed(1)+'%).</div>';}
 out.innerHTML=
   stat('', (pOver*100).toFixed(1)+'%','model P(over '+line+')')+
   stat('', fmtOdds(pOver),'fair over odds')+
   stat('', (mOver!=null?(mOver*100).toFixed(1)+'%':'–'),'market P(over), de-vigged')+
   stat('', (push>0.001?(push*100).toFixed(1)+'%':'0%'),'push chance')+
   stat(best[0]==='OVER'?'win':'', (evOver!=null?(evOver*100).toFixed(1)+'%':'–'),'EV backing over')+
   stat(best[0]==='UNDER'?'win':'', (evUnder!=null?(evUnder*100).toFixed(1)+'%':'–'),'EV backing under')+
   verdict;
}
function drawCurve(mu,sd,line){
 var W=520,H=240,ml=8,mr=8,mt=10,mb=22,pw=W-ml-mr,ph=H-mt-mb;
 var x0=mu-3.5*sd,x1=mu+3.5*sd;
 var sx=function(x){return ml+(x-x0)/(x1-x0)*pw;};
 var pdf=function(x){return Math.exp(-0.5*Math.pow((x-mu)/sd,2));};
 var top=pdf(mu),sy=function(v){return mt+(1-v/top)*ph;};
 var N=120,pts=[],fill=[];
 for(var i=0;i<=N;i++){var x=x0+(x1-x0)*i/N;pts.push([sx(x),sy(pdf(x))]);}
 for(var j=0;j<=N;j++){var x=Math.max(line,x0)+(x1-Math.max(line,x0))*j/N;fill.push([sx(x),sy(pdf(x))]);}
 var path='M'+pts.map(p=>p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' L');
 var area='M'+sx(Math.max(line,x0)).toFixed(1)+' '+(mt+ph)+' L'+
   fill.map(p=>p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' L')+' L'+sx(x1).toFixed(1)+' '+(mt+ph)+' Z';
 var lx=sx(line);
 var svg='<svg viewBox="0 0 '+W+' '+H+'" class="chart">'+
  '<path d="'+area+'" fill="#39d98a" opacity="0.18"/>'+
  '<path d="'+path+'" fill="none" stroke="#4cc2ff" stroke-width="2"/>'+
  '<line x1="'+lx.toFixed(1)+'" y1="'+mt+'" x2="'+lx.toFixed(1)+'" y2="'+(mt+ph)+'" stroke="#f0a35e" stroke-width="1.5" stroke-dasharray="4 3"/>'+
  '<text x="'+lx.toFixed(1)+'" y="'+(mt+ph+15)+'" text-anchor="middle" fill="#f0a35e" font-size="11">line '+line.toFixed(1)+'</text>'+
  '<text x="'+sx(mu).toFixed(1)+'" y="'+(mt+12)+'" text-anchor="middle" fill="#8aa0b2" font-size="11">model '+mu.toFixed(1)+'</text>'+
  '<text x="'+(ml+pw-4)+'" y="'+(mt+ph-6)+'" text-anchor="end" fill="#39d98a" font-size="11">over →</text></svg>';
 document.getElementById('lab-curve').innerHTML=svg;
}

// ---- Compare dashboard filters ----
function cmpFilter(){
 var tbl=document.getElementById('cmp'); if(!tbl)return;
 var match=(document.getElementById('f-match')||{}).value||'all';
 var market=(document.getElementById('f-market')||{}).value||'all';
 var evonly=(document.getElementById('f-ev')||{}).checked;
 var cred=(document.getElementById('f-cred')||{}).checked;
 var shown=0;
 tbl.querySelectorAll('tbody tr').forEach(function(tr){
   var ok=true, ev=parseFloat(tr.dataset.ev);
   if(match!=='all' && tr.dataset.match!==match) ok=false;
   if(market!=='all' && tr.dataset.market!==market) ok=false;
   if(evonly && !(ev>0)) ok=false;
   if(cred && (!isNaN(ev) && (ev>40||ev<-95))) ok=false;  // hide implausible longshots
   tr.style.display=ok?'':'none'; if(ok)shown++;
 });
 var c=document.getElementById('f-count'); if(c)c.textContent=shown+' markets';
}
// ---- Compare: manual price -> EV vs model (for books we can't pull live, e.g. Dabble) ----
function cmpManual(inp){
 var rk=inp.dataset.rk;
 var cell=document.querySelector('.mev[data-rk="'+(window.CSS&&CSS.escape?CSS.escape(rk):rk)+'"]');
 var myp=parseFloat(inp.dataset.myp), price=parseFloat(inp.value);
 try{ price>0 ? localStorage.setItem('cmpmp:'+rk, inp.value) : localStorage.removeItem('cmpmp:'+rk); }catch(e){}
 if(!cell) return;
 if(!(price>0)){ cell.textContent=''; cell.className='mev'; return; }
 if(!(myp>0)){ cell.textContent='?'; cell.className='mev mut'; return; }
 var ev=myp*price-1;
 cell.textContent=(ev>=0?'+':'')+(ev*100).toFixed(0)+'%';
 cell.className='mev '+(ev>0?'pos':'neg');
}
function cmpRestore(){
 document.querySelectorAll('#cmp input.mp').forEach(function(inp){
   var v=null; try{ v=localStorage.getItem('cmpmp:'+inp.dataset.rk); }catch(e){}
   if(v){ inp.value=v; cmpManual(inp); }
 });
}
// ---- Pick'em: client-side model maths (mirrors pricing.py / player_points.py) ----
function _erf(x){var s=x<0?-1:1;x=Math.abs(x);var t=1/(1+0.3275911*x);
 var y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);
 return s*y;}
function _ncdf(z){return 0.5*(1+_erf(z/Math.SQRT2));}
function _poisPmf(lam,k){var p=Math.exp(-lam);for(var i=1;i<=k;i++)p*=lam/i;return p;}
function _poisTail(lam,line){var fl=Math.floor(line),s=0;for(var k=0;k<=fl;k++)s+=_poisPmf(lam,k);return 1-s;}
function _pointsPover(lt,lg,lfg,line){ // 4*Pois(lt)+2*Pois(lg)+Pois(lfg), kmax 8/14/4
 var T=[],G=[],F=[],i;lt=Math.max(lt,1e-9);lg=Math.max(lg,1e-9);lfg=Math.max(lfg,1e-9);
 for(i=0;i<=8;i++)T.push(_poisPmf(lt,i));for(i=0;i<=14;i++)G.push(_poisPmf(lg,i));for(i=0;i<=4;i++)F.push(_poisPmf(lfg,i));
 var s=0;for(var ti=0;ti<T.length;ti++)for(var gi=0;gi<G.length;gi++){var base=4*ti+2*gi,ptg=T[ti]*G[gi];
   for(var fi=0;fi<F.length;fi++){if(base+fi>line)s+=ptg*F[fi];}}
 return s;}
function _normOver(mu,sg,line){sg=Math.max(sg,1e-6);var f=Math.round((line-Math.floor(line))*1e4)/1e4;
 if(f===0.25||f===0.75)return (_normOver(mu,sg,line-0.25)+_normOver(mu,sg,line+0.25))/2;
 if(Math.abs(f-0.5)<1e-6)return 1-_ncdf((line-mu)/sg);
 return 1-_ncdf((line+0.5-mu)/sg);} // integer line: push band, p_over = 1-cdf(line+0.5)
function pkPover(d,line){ if(!d||isNaN(line))return null;
 if(d.k==='pois')return _poisTail(d.lam,line);
 if(d.k==='conv')return _pointsPover(d.lt,d.lg,d.lfg,line);
 if(d.k==='norm')return _normOver(d.mu,d.sg,line); return null;}
function _fair(p){return (p&&p>1e-9)?('$'+(1/p).toFixed(2)):'$–';}
function pkCompute(tr){
 var d; try{d=JSON.parse(tr.dataset.dist);}catch(e){return;}
 var line=parseFloat(tr.querySelector('input.lnin').value);
 var ov=tr.querySelector('.pko'), un=tr.querySelector('.pku');
 var po=pkPover(d,line);
 if(po===null){ov.innerHTML='';un.innerHTML='';tr.dataset.p=0;return;}
 po=Math.min(Math.max(po,0),1);var pu=1-po;
 tr.dataset.po=po;tr.dataset.line=line;tr.dataset.p=Math.max(po,pu);
 ov.className='pko'+(po>=0.6?' pos':'');un.className='pku'+(pu>=0.6?' pos':'');
 ov.innerHTML='<b>'+(po*100).toFixed(0)+'%</b> <span class="mut">'+_fair(po)+'</span> <button class="addleg" data-sd="over" onclick="pkAdd(this)">+</button>';
 un.innerHTML='<b>'+(pu*100).toFixed(0)+'%</b> <span class="mut">'+_fair(pu)+'</span> <button class="addleg" data-sd="under" onclick="pkAdd(this)">+</button>';
}
function pkComputeAll(){ var t=document.getElementById('pkm'); if(t)t.querySelectorAll('tbody tr').forEach(pkCompute); }
function pkFilter(){
 var tbl=document.getElementById('pkm'); if(!tbl)return;
 var match=(document.getElementById('pk-match')||{}).value||'all';
 var stat=(document.getElementById('pk-stat')||{}).value||'all';
 var strong=(document.getElementById('pk-strong')||{}).checked;
 var n=0;
 tbl.querySelectorAll('tbody tr').forEach(function(tr){
   var ok=true, p=parseFloat(tr.dataset.p);
   if(match!=='all' && tr.dataset.match!==match) ok=false;
   if(stat!=='all' && tr.dataset.stat!==stat) ok=false;
   if(strong && !(p>=0.6)) ok=false;
   tr.style.display=ok?'':'none'; if(ok)n++;
 });
 var c=document.getElementById('pk-count'); if(c)c.textContent=n+' rows';
}
var PK_SLIP=[];
function pkAdd(btn){
 var tr=btn.closest('tr'); if(!tr.dataset.po) pkCompute(tr);
 var side=btn.dataset.sd, po=parseFloat(tr.dataset.po);
 if(isNaN(po))return;
 var p=side==='over'?po:1-po, ln=parseFloat(tr.dataset.line);
 var pl=tr.dataset.pl, st=tr.dataset.stat;
 if(PK_SLIP.find(function(l){return l.pl===pl&&l.st===st&&l.ln===ln&&l.sd===side;}))return;
 PK_SLIP.push({pl:pl,st:st,ln:ln,sd:side,p:p,btn:btn});
 btn.textContent='✓'; btn.disabled=true; renderSlip();
}
function rmLeg(i){ var l=PK_SLIP[i]; if(l&&l.btn){l.btn.textContent='+';l.btn.disabled=false;} PK_SLIP.splice(i,1); renderSlip(); }
function clearSlip(){ PK_SLIP.forEach(function(l){if(l.btn){l.btn.textContent='+';l.btn.disabled=false;}}); PK_SLIP=[]; renderSlip(); }
function renderSlip(){
 var el=document.getElementById('slip'); if(!el)return;
 if(!PK_SLIP.length){el.className='slip';el.textContent='Slip empty — add legs to build a parlay.';return;}
 var prod=PK_SLIP.reduce(function(a,l){return a*l.p;},1);
 var n=PK_SLIP.length; var m=(window.PK_MULT||{})[n];
 var legsHtml=PK_SLIP.map(function(l,i){return '<span class="chip">'+l.pl+' '+l.sd.toUpperCase()+' '+l.ln+' ('+(l.p*100).toFixed(0)+'%) <a onclick="rmLeg('+i+')">×</a></span>';}).join(' ');
 var out='<div class="sliphead"><b>'+n+'-leg parlay</b> <button class="clearbtn" onclick="clearSlip()">Clear</button></div>'+legsHtml+'<div class="slipres">';
 if(!m){ out+= n<2 ? 'Add at least 2 legs (minimum for a parlay).' : 'No multiplier for '+n+' legs.'; }
 else { var ev=m*prod-1; out+='combined win prob <b>'+(prod*100).toFixed(1)+'%</b> · multiplier <b>×'+m+'</b> · theoretical EV <b class="'+(ev>0?'pos':'neg')+'">'+(ev*100>=0?'+':'')+(ev*100).toFixed(0)+'%</b>'; }
 out+='</div>';
 el.className='slip on'; el.innerHTML=out;
}
// ---- Scoring match filter ----
function scFilter(match){
 document.querySelectorAll('section.match').forEach(function(s){
   s.style.display=(match==='all'||s.dataset.match===match)?'':'none';});
}
// ---- Tabs (Scoring) ----
function showTab(group,name){
 document.querySelectorAll('[data-tabgroup="'+group+'"]').forEach(function(b){
   b.classList.toggle('on', b.dataset.tab===name);});
 document.querySelectorAll('[data-pane="'+group+'"]').forEach(function(p){
   p.classList.toggle('on', p.dataset.paneName===name);});
}
document.addEventListener('DOMContentLoaded', function(){
 if(document.getElementById('cmp')){ cmpFilter(); cmpRestore(); }
 if(document.getElementById('pkm')){ pkComputeAll(); pkFilter(); }
});

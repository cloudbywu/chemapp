'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {captureWorkspace}=require('../capture-workspace.cjs');
test('workspace export writes captured PNG only to the user-chosen file',async()=>{
  const bytes=Buffer.from('png-test'); const writes=[];
  const result=await captureWorkspace({contents:{capturePage:async()=>({toPNG:()=>bytes})},choosePath:async()=>({canceled:false,filePath:'/chosen/image.png'}),writeFile:async(...args)=>writes.push(args)});
  assert.equal(result,true);assert.deepEqual(writes,[['/chosen/image.png',bytes]]);
});
test('cancelled screenshot export does not write a file',async()=>{
  let writes=0;
  assert.equal(await captureWorkspace({contents:{capturePage:async()=>({toPNG:()=>Buffer.from('x')})},choosePath:async()=>({canceled:true}),writeFile:async()=>writes++}),false);
  assert.equal(writes,0);
});
test('capture failure cannot create an empty file or request a destination',async()=>{
  let selected=false;
  await assert.rejects(captureWorkspace({contents:{capturePage:async()=>{throw new Error('not ready');}},choosePath:async()=>{selected=true;},writeFile:async()=>{throw new Error('unexpected');}}),/not ready/);
  assert.equal(selected,false);
});

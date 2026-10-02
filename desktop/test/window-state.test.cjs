'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { initialWindowState, requestQuit } = require('../window-state.cjs');
const area = {x:0,y:0,width:1364,height:960};
test('first window and invalid saved bounds stay within current work area', () => {
  const state = initialWindowState({},area);
  assert.equal(state.width,1316); assert.equal(state.height,912);
  for(const saved of [{x:-9999,y:9999,width:9000,height:NaN},{x:Infinity,width:-1,height:2},null]) {
    const result = initialWindowState(saved,area);
    assert(result.x>=0 && result.x+result.width<=area.width);
    assert(result.y>=0 && result.y+result.height<=area.height);
  }
});
test('small displays do not create unreachable window edges', () => {
  const result=initialWindowState({}, {x:-800,y:0,width:800,height:600});
  assert.equal(result.width,800);assert.equal(result.height,600);
  assert.equal(result.x,-800);assert.equal(result.minHeight,600);
});
test('quit first requests renderer close and never stops a cancelled window backend', () => {
  let prevented=0,closed=0,stopped=0;
  const event={preventDefault(){prevented++;}};
  const window={isDestroyed:()=>false,close(){closed++;}};
  requestQuit({event,window,shutdown(){stopped++;},alreadyQuitting:false});
  assert.equal(prevented,1);assert.equal(closed,1);assert.equal(stopped,0);
});
test('backend shutdown only begins after close; final quit does not loop', () => {
  let prevented=0,stopped=0;
  const event={preventDefault(){prevented++;}};
  requestQuit({event,window:{isDestroyed:()=>true},shutdown(){stopped++;},alreadyQuitting:false});
  assert.equal(stopped,1);
  requestQuit({event,window:null,shutdown(){stopped++;},alreadyQuitting:true});
  assert.equal(prevented,1);assert.equal(stopped,1);
});

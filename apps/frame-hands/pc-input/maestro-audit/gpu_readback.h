#pragma once
// Diagnostic readback of the acquired native view. Does not alter image contents.
struct GpuReadback {
 VkDevice device{}; VkBuffer buffer{}; VkDeviceMemory memory{};
 VkCommandPool pool{}; VkCommandBuffer command{}; VkFence fence{};
 VkDeviceSize capacity{}; void* mapped{};
 ~GpuReadback(){
  if(!device)return;
  vkDeviceWaitIdle(device);
  if(mapped)vkUnmapMemory(device,memory);
  if(fence)vkDestroyFence(device,fence,nullptr);
  if(pool)vkDestroyCommandPool(device,pool,nullptr);
  if(buffer)vkDestroyBuffer(device,buffer,nullptr);
  if(memory)vkFreeMemory(device,memory,nullptr);
 }
 void read(const vr::VROverlayView_t& view,std::vector<unsigned char>& pixels,uint32_t& w,uint32_t& h){
  require(view.texture.eType==vr::TextureType_Vulkan&&view.texture.handle,"No GPU headset image");
  const auto& t=*static_cast<const vr::VRVulkanTextureData_t*>(view.texture.handle);
  require(t.m_nWidth&&t.m_nHeight&&uint64_t(t.m_nWidth)*t.m_nHeight<=16000000,"GPU image dimensions");
  require(t.m_nFormat==VK_FORMAT_R8G8B8A8_UNORM||t.m_nFormat==VK_FORMAT_R8G8B8A8_SRGB||t.m_nFormat==VK_FORMAT_B8G8R8A8_UNORM||t.m_nFormat==VK_FORMAT_B8G8R8A8_SRGB,"GPU image format");
  const VkDeviceSize bytes=uint64_t(t.m_nWidth)*t.m_nHeight*4;
  if(!device){
   device=t.m_pDevice;capacity=bytes;
   fprintf(stdout,"GPU image %ux%u format=%u samples=%u\n",t.m_nWidth,t.m_nHeight,t.m_nFormat,t.m_nSampleCount);fflush(stdout);
   VkBufferCreateInfo b{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};b.size=bytes;b.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;b.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
   checked(vkCreateBuffer(device,&b,nullptr,&buffer),"readback buffer");
   VkMemoryRequirements requirements{};vkGetBufferMemoryRequirements(device,buffer,&requirements);
   VkPhysicalDeviceMemoryProperties properties{};vkGetPhysicalDeviceMemoryProperties(t.m_pPhysicalDevice,&properties);
   uint32_t type=0;const auto flags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for(;type<properties.memoryTypeCount;++type)if((requirements.memoryTypeBits&(1u<<type))&&(properties.memoryTypes[type].propertyFlags&flags)==flags)break;
   require(type<properties.memoryTypeCount,"coherent readback memory");
   VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};allocation.allocationSize=requirements.size;allocation.memoryTypeIndex=type;
   checked(vkAllocateMemory(device,&allocation,nullptr,&memory),"readback allocation");checked(vkBindBufferMemory(device,buffer,memory,0),"readback bind");checked(vkMapMemory(device,memory,0,bytes,0,&mapped),"readback map");
   VkCommandPoolCreateInfo p{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};p.queueFamilyIndex=t.m_nQueueFamilyIndex;p.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
   checked(vkCreateCommandPool(device,&p,nullptr,&pool),"readback pool");
   VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};a.commandPool=pool;a.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;a.commandBufferCount=1;
   checked(vkAllocateCommandBuffers(device,&a,&command),"readback commands");
   VkFenceCreateInfo f{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};checked(vkCreateFence(device,&f,nullptr,&fence),"readback fence");
  }
  require(device==t.m_pDevice&&bytes<=capacity,"GPU view changed allocation");
  checked(vkResetCommandBuffer(command,0),"reset readback");
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;checked(vkBeginCommandBuffer(command,&begin),"begin readback");
  VkImage image=(VkImage)t.m_nImage;
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
  barrier.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;barrier.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;barrier.image=image;barrier.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
  vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
  VkBufferImageCopy copy{};copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};copy.imageExtent={t.m_nWidth,t.m_nHeight,1};
  vkCmdCopyImageToBuffer(command,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffer,1,&copy);
  VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};host.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;host.srcQueueFamilyIndex=host.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;host.buffer=buffer;host.size=bytes;
  vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&host,0,nullptr);
  checked(vkEndCommandBuffer(command),"end readback");checked(vkResetFences(device,1,&fence),"reset readback fence");
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;
  checked(vkQueueSubmit(t.m_pQueue,1,&submit,fence),"submit readback");checked(vkWaitForFences(device,1,&fence,VK_TRUE,3000000000ull),"wait readback");
  w=t.m_nWidth;h=t.m_nHeight;pixels.resize(bytes);memcpy(pixels.data(),mapped,bytes);
  if(t.m_nFormat==VK_FORMAT_B8G8R8A8_UNORM||t.m_nFormat==VK_FORMAT_B8G8R8A8_SRGB)for(size_t k=0;k<pixels.size();k+=4)std::swap(pixels[k],pixels[k+2]);
 }
};
